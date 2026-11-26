import json
import logging
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import yaml

pytest.importorskip("torch")

from sleep_classifier import raw_session_cli  # noqa: E402
from sleep_classifier.raw_session_cli import SessionCLIConfig, main, run_stream  # noqa: E402


class RecordingInput(BytesIO):
    """Record every requested byte limit for a synthetic input stream."""

    def __init__(self, data: bytes) -> None:
        """Load input bytes and initialize the read history.

        Args:
            data (bytes): Synthetic JSONL or invalid bytes.
        """
        super().__init__(data)
        self.read_sizes: list[int] = []

    def readline(self, size: int = -1) -> bytes:
        """Record the limit and read one line.

        Args:
            size (int): Maximum number of bytes to read.
        """
        self.read_sizes.append(size)
        return super().readline(size)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Write a configuration with a relative bundle path containing spaces.

    Args:
        tmp_path (Path): Temporary workspace for synthetic inputs.
    """
    directory = tmp_path.resolve() / "session configs"
    directory.mkdir()
    config = directory / "raw session.yaml"
    config.write_text(
        "bundle_dir: path with spaces/evaluation bundle\n"
        "max_buffered_events: 12\nmax_epochs_per_event: 3\nmax_line_bytes: 256\n",
        encoding="utf-8",
    )
    return config


@pytest.mark.parametrize("field", ["max_buffered_events", "max_epochs_per_event", "max_line_bytes"])
@pytest.mark.parametrize("value", [True, 0, -1, 1.0, "1"])
def test_config_rejects_nonpositive_or_noninteger_limits(
    config_path: Path, field: str, value: bool | int | float | str,
) -> None:
    """Reject invalid limits, including YAML booleans and numeric strings.

    Args:
        config_path (Path): Configuration to overwrite.
        field (str): Limit to make invalid.
        value (bool | int | float | str): Invalid setting value.
    """
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings[field] = value
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        SessionCLIConfig.load(config_path)


@pytest.mark.parametrize(
    "change", [{"unsafe_runtime": True}, {"max_line_bytes": None}, {"bundle_dir": " "}],
)
def test_config_requires_supported_settings(
    config_path: Path, change: dict[str, bool | str | None],
) -> None:
    """Reject unsupported settings, missing settings, and empty bundle paths.

    Args:
        config_path (Path): Configuration to overwrite.
        change (dict[str, bool | str | None]): Setting to replace or remove.
    """
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings.update(change)
    settings = {name: value for name, value in settings.items() if value is not None}
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with pytest.raises(ValueError):
        SessionCLIConfig.load(config_path)


def test_max_line_bytes_includes_newline_and_precedes_decoding() -> None:
    """Accept a boundary-sized line and stop on the next oversized line."""
    source = RecordingInput(b"{}\n\xff12\n{}\n")
    runner = Mock()
    runner.handle.return_value = []

    with pytest.raises(ValueError, match=r"Line 2: exceeds max_line_bytes \(3\)"):
        run_stream(source, StringIO(), runner, max_line_bytes=3)

    assert source.read_sizes == [4, 4]
    runner.handle.assert_called_once_with({})
    runner.ensure_finished.assert_not_called()


@pytest.mark.parametrize(
    "bad_line, reason",
    [
        (b"\xff\n", "expected UTF-8"),
        (b" \t\r\n", "blank lines are invalid"),
        (b'{"secret":\n', "invalid JSON event"),
        (b'{"secret":1,"secret":2}\n', "invalid JSON event"),
        (b'{"secret":NaN}\n', "invalid JSON event"),
        (b'{"secret":1e1000}\n', "invalid JSON event"),
    ],
)
def test_stream_stops_at_first_invalid_input(bad_line: bytes, reason: str) -> None:
    """Reject invalid input without including event contents or reading later lines.

    Args:
        bad_line (bytes): Invalid line to place before a valid JSON record.
        reason (str): Diagnostic without raw data.
    """
    source = RecordingInput(bad_line + b"{}\n")
    runner = Mock()

    with pytest.raises(ValueError, match=f"^Line 1: {reason}$"):
        run_stream(source, StringIO(), runner, max_line_bytes=256)

    assert source.read_sizes == [257]
    runner.handle.assert_not_called()
    runner.ensure_finished.assert_not_called()


def test_runner_rejection_is_line_indexed_and_redacted() -> None:
    """Stop after a rejected event without reporting its contents."""
    source = RecordingInput(b'{"secret":"do not log"}\n{}\n')
    runner = Mock()
    runner.handle.side_effect = ValueError("secret: do not log")

    with pytest.raises(ValueError, match=r"^Line 1: session event rejected$"):
        run_stream(source, StringIO(), runner, max_line_bytes=256)

    assert source.read_sizes == [257]
    runner.handle.assert_called_once_with({"secret": "do not log"})
    runner.ensure_finished.assert_not_called()


def test_stream_requires_explicit_finish() -> None:
    """Reject EOF when the runner has not accepted a finish event."""
    runner = Mock()
    runner.handle.return_value = []
    runner.ensure_finished.side_effect = RuntimeError("missing finish")

    with pytest.raises(ValueError, match=r"^Line 2: reached EOF before session finish$"):
        run_stream(BytesIO(b'{"type":"start"}\n'), StringIO(), runner, max_line_bytes=256)

    runner.ensure_finished.assert_called_once_with()


def test_stream_flushes_each_record_and_omits_empty_batches() -> None:
    """Flush output after each record and accept a final line without a newline."""
    runner = Mock()
    records = [
        [{"type": "session_started"}],
        [],
        [{"type": "prediction", "probabilities": [0.25] * 4}, {"type": "skipped_epoch"}],
        [{"type": "session_finished"}],
    ]
    runner.handle.side_effect = records
    types = ("start", "heart_rate", "watermark", "finish")
    events = [{"type": kind} for kind in types]
    source = BytesIO("\n".join(map(json.dumps, events)).encode("utf-8"))
    output = Mock()

    run_stream(source, output, runner, max_line_bytes=256)

    assert runner.handle.call_args_list == [call(event) for event in events]
    expected_calls = [
        operation
        for batch in records for record in batch
        for operation in (call.write(json.dumps(record) + "\n"), call.flush())
    ]
    assert output.mock_calls == expected_calls
    runner.ensure_finished.assert_called_once_with()


def test_stream_refuses_nonfinite_output() -> None:
    """Reject a nonfinite runner record before writing any bytes for it."""
    runner = Mock()
    runner.handle.return_value = [{"score": float("nan")}]
    output = StringIO()

    with pytest.raises(ValueError, match="Out of range float"):
        run_stream(BytesIO(b"{}\n"), output, runner, max_line_bytes=256)

    assert output.getvalue() == ""
    runner.ensure_finished.assert_not_called()


def test_main_rejects_runtime_mismatch_before_opening_input(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report bundle verification failure before trying to open the absent input file.

    Args:
        config_path (Path): Valid synthetic configuration.
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing dependency boundaries.
        caplog (pytest.LogCaptureFixture): Captured operational log records.
        capsys (pytest.CaptureFixture[str]): Captured standard output and error.
    """
    make_bundle = Mock(side_effect=ValueError("Bundle runtime source_revision does not match"))
    make_runner = Mock()
    monkeypatch.setattr(raw_session_cli, "EvaluationBundle", make_bundle)
    monkeypatch.setattr(raw_session_cli, "RawSessionRunner", make_runner)

    with pytest.raises(SystemExit) as error:
        main(["--config", str(config_path), "--input", str(config_path.parent / "missing data")])

    assert error.value.code == 1
    make_bundle.assert_called_once_with(config_path.parent / "path with spaces/evaluation bundle")
    make_runner.assert_not_called()
    assert "Bundle runtime source_revision does not match" in caplog.text
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("use_stdin", [False, True])
def test_main_loads_one_bundle_and_writes_only_protocol_records(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str], use_stdin: bool,
) -> None:
    """Use one verified bundle with configured limits and input from a file or stdin.

    Args:
        config_path (Path): Valid synthetic configuration.
        monkeypatch (pytest.MonkeyPatch): Fixture for replacing dependency boundaries.
        caplog (pytest.LogCaptureFixture): Captured operational log records.
        capsys (pytest.CaptureFixture[str]): Captured standard output and error.
        use_stdin (bool): Select default stdin instead of a path with spaces.
    """
    make_bundle = Mock()
    make_runner = Mock()
    payload = b'{"type":"finish"}\n'
    record = {"type": "session_finished", "predicted_epochs": 7}
    make_runner.return_value.handle.return_value = [record]
    monkeypatch.setattr(raw_session_cli, "EvaluationBundle", make_bundle)
    monkeypatch.setattr(raw_session_cli, "RawSessionRunner", make_runner)
    args = ["--config", str(config_path)]
    with TextIOWrapper(BytesIO(payload)) as stdin:
        if use_stdin:
            monkeypatch.setattr(raw_session_cli.sys, "stdin", stdin)
        else:
            input_path = config_path.parent / "raw records.jsonl"
            input_path.write_bytes(payload)
            args.extend(["--input", str(input_path)])

        with caplog.at_level(logging.INFO):
            main(args)

        assert not stdin.closed

    make_bundle.assert_called_once_with(config_path.parent / "path with spaces/evaluation bundle")
    make_runner.assert_called_once_with(
        make_bundle.return_value, max_buffered_events=12, max_epochs_per_event=3,
    )
    make_runner.return_value.handle.assert_called_once_with({"type": "finish"})
    make_runner.return_value.ensure_finished.assert_called_once_with()
    assert capsys.readouterr().out.splitlines() == [json.dumps(record)]
    assert "Verified evaluation bundle" in caplog.text
    assert "Session finished" in caplog.text
