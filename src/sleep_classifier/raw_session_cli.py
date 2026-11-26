import argparse
import json
import logging
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import yaml

from sleep_classifier.bundle_loader import EvaluationBundle
from sleep_classifier.bundle_validation import strict_json_loads
from sleep_classifier.raw_session_runner import RawSessionRunner

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionCLIConfig:
    """Bundle location and resource limits for the JSONL entry point."""

    bundle_dir: Path
    max_buffered_events: int
    max_epochs_per_event: int
    max_line_bytes: int

    @classmethod
    def load(cls, path: Path) -> "SessionCLIConfig":
        """Read the supported settings and resolve the bundle path relative to the config.

        Args:
            path (Path): Session configuration YAML file.
        """
        path = path.resolve()
        try:
            settings = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (UnicodeError, yaml.YAMLError) as error:
            raise ValueError("Session config must contain valid UTF-8 YAML") from error
        limits = ("max_buffered_events", "max_epochs_per_event", "max_line_bytes")
        if not isinstance(settings, dict) or settings.keys() != {"bundle_dir", *limits}:
            raise ValueError("Session config must contain exactly the documented settings")
        bundle_dir = settings["bundle_dir"]
        if type(bundle_dir) is not str or not bundle_dir.strip():
            raise ValueError("bundle_dir must be a nonempty string")
        for name in limits:
            if type(settings[name]) is not int or settings[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return cls(
            bundle_dir=path.parent / bundle_dir,
            max_buffered_events=settings["max_buffered_events"],
            max_epochs_per_event=settings["max_epochs_per_event"],
            max_line_bytes=settings["max_line_bytes"],
        )


def _decode_event(line: bytes, line_number: int) -> Any:
    """Decode one input event without exposing its values in error messages.

    Args:
        line (bytes): Bounded event line as read from the input.
        line_number (int): One-based line position.
    """
    try:
        data = line.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"Line {line_number}: expected UTF-8") from error
    if not data.strip():
        raise ValueError(f"Line {line_number}: blank lines are invalid")
    try:
        return strict_json_loads(data)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"Line {line_number}: invalid JSON event") from error


def run_stream(
    input_stream: BinaryIO, output_stream: TextIO, runner: RawSessionRunner, max_line_bytes: int,
) -> None:
    """Read bounded JSONL events and flush each resulting record in order.

    Args:
        input_stream (BinaryIO): Binary stream with one UTF-8 event per line.
        output_stream (TextIO): Text stream for protocol records.
        runner (RawSessionRunner): Runner backed by a verified bundle.
        max_line_bytes (int): Maximum line size in bytes, including any newline.
    """
    if type(max_line_bytes) is not int or max_line_bytes <= 0:
        raise ValueError("max_line_bytes must be a positive integer")
    read_size = min(max_line_bytes + 1, sys.maxsize)
    line_number = 1
    while True:
        try:
            line = input_stream.readline(read_size)
        except OSError as error:
            raise ValueError(f"Line {line_number}: cannot read input") from error
        if not line:
            break
        if len(line) > max_line_bytes:
            raise ValueError(f"Line {line_number}: exceeds max_line_bytes ({max_line_bytes})")
        event = _decode_event(line, line_number)
        try:
            records = runner.handle(event)
        except BufferError as error:
            raise ValueError(f"Line {line_number}: session buffer limit exceeded") from error
        except (ValueError, RuntimeError, OSError) as error:
            raise ValueError(f"Line {line_number}: session event rejected") from error
        for record in records:
            output_stream.write(json.dumps(record, allow_nan=False) + "\n")
            output_stream.flush()
        line_number += 1

    try:
        runner.ensure_finished()
    except (ValueError, RuntimeError) as error:
        raise ValueError(f"Line {line_number}: reached EOF before session finish") from error


def main(argv: list[str] | None = None) -> None:
    """Run a raw session against one verified evaluation bundle.

    Args:
        argv (list[str] | None): Arguments to parse. Defaults to process arguments.
    """
    parser = argparse.ArgumentParser(description="Run a raw session with a verified bundle")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", default="-", help="JSONL input file or '-' for standard input")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        config = SessionCLIConfig.load(args.config)
        bundle = EvaluationBundle(config.bundle_dir)
        runner = RawSessionRunner(
            bundle, max_buffered_events=config.max_buffered_events,
            max_epochs_per_event=config.max_epochs_per_event,
        )
        LOGGER.info("Verified evaluation bundle at %s", config.bundle_dir)
        with ExitStack() as stack:
            input_stream = (
                sys.stdin.buffer if args.input == "-"
                else stack.enter_context(Path(args.input).open("rb"))
            )
            run_stream(input_stream, sys.stdout, runner, config.max_line_bytes)
        LOGGER.info("Session finished")
    except (ValueError, RuntimeError, BufferError, OSError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
