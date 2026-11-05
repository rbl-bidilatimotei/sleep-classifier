import json
from pathlib import Path

import pytest

from sleep_classifier.opal_report import OpalRunReport


def _write_run(root: Path, events: list[dict], response: dict | None = None) -> None:
    opal = root / "opal"
    opal.mkdir(parents=True)
    (opal / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "best_epoch": 1,
                "optimizer": {"updates": 1872, "agent": {"backend": "codex", "max_tokens": 20000}},
                "history": [{}] * 6,
            }
        )
    )
    if response is not None:
        directory = opal / "decisions/001"
        directory.mkdir(parents=True)
        (directory / "response.json").write_text(json.dumps(response))


def test_failed_backend_is_not_a_successful_live_comparison(tmp_path):
    usage = {
        "input_tokens": 178791,
        "cached_input_tokens": 154368,
        "output_tokens": 4878,
        "reasoning_output_tokens": 1083,
    }
    _write_run(
        tmp_path,
        [
            {"kind": "start", "backend": "codex", "schedule": [350, 3300]},
            {"kind": "requested", "k": 1, "update": 350},
            {"kind": "rejected", "k": 1, "failed": True},
            {
                "kind": "end",
                "update": 1872,
                "version": 0,
                "agent_wait_seconds": 131.3,
                "usage": usage,
                "cost_usd": None,
                "unknown_cost_decisions": [1],
            },
        ],
        {
            "k": 1,
            "backend_error": "Unexpected provider non-message item: additional_tools",
            "usage": usage,
            "text": "SECRET PROMPT",
            "session": "DO NOT DISPLAY",
        },
    )
    report = OpalRunReport(tmp_path).build()
    assert report["status"] == "failed_no_changes_applied"
    assert report["decision_counts"]["applied"] == 0
    assert report["usage"]["total_tokens"] == 183669
    assert report["soft_token_cap_exceeded"] is True
    assert report["usage"]["uncached_input_tokens"] == 24423
    assert report["usage"]["source"] == "end_event"
    assert report["unreached_scheduled_updates"] == [3300]
    assert report["selected_checkpoint_update"] == 312
    assert "SECRET PROMPT" not in json.dumps(report)
    assert "DO NOT DISPLAY" not in json.dumps(report)
    assert report["cost_usd"] is None


def test_applied_change_and_reversion_remain_visible(tmp_path):
    _write_run(
        tmp_path,
        [
            {"kind": "start", "backend": "codex", "schedule": [350]},
            {"kind": "requested", "k": 1, "update": 350},
            {"kind": "applied", "k": 1, "version": 1, "applied_at": 350},
            {"kind": "reverted", "version": 1, "update": 360},
            {"kind": "end", "update": 1872, "version": 0},
        ],
    )
    report = OpalRunReport(tmp_path).build()
    assert report["status"] == "changes_applied_with_reversions"
    assert report["decision_counts"]["applied"] == 1
    assert report["decision_counts"]["reverted"] == 1
    assert report["checkpoint_precedes_first_change"] is True


def test_offline_keep_is_distinct_from_live_success(tmp_path):
    _write_run(
        tmp_path,
        [
            {"kind": "start", "backend": "keep", "schedule": [350]},
            {"kind": "kept", "k": 1},
            {"kind": "end", "version": 0, "update": 1872},
        ],
    )
    assert OpalRunReport(tmp_path).build()["status"] == "offline_keep_control"


def test_missing_and_partial_logs_are_not_success(tmp_path):
    report = OpalRunReport(tmp_path).build()
    assert report["status"] == "unknown_missing_events"
    assert report["decision_counts"] is None
    assert report["usage"]["input_tokens"] is None
    _write_run(tmp_path, [{"kind": "start", "backend": "codex", "schedule": [350]}])
    assert OpalRunReport(tmp_path).build()["status"] == "incomplete"


def test_response_usage_is_used_only_when_end_usage_is_missing(tmp_path):
    _write_run(
        tmp_path,
        [{"kind": "start", "backend": "codex"}],
        {
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 7,
                "output_tokens": 2,
                "reasoning_output_tokens": 1,
            }
        },
    )
    usage = OpalRunReport(tmp_path).build()["usage"]
    assert usage["source"] == "decision_responses"
    assert usage["total_tokens"] == 12


@pytest.mark.parametrize(
    "filename,content",
    [
        ("events.jsonl", '{"kind":"start"}\n{'),
        ("events.jsonl", "[]"),
        ("decisions/001/response.json", "[]"),
    ],
)
def test_corrupt_metadata_raises_instead_of_becoming_success(tmp_path, filename, content):
    path = tmp_path / "opal" / filename
    path.parent.mkdir(parents=True)
    path.write_text(content)
    with pytest.raises(ValueError, match="Invalid|Expected"):
        OpalRunReport(tmp_path).build()


def test_applied_change_is_reported_without_claiming_a_score_gain(tmp_path):
    _write_run(
        tmp_path,
        [
            {"kind": "start", "backend": "codex", "schedule": [100]},
            {"kind": "requested", "k": 1, "update": 100},
            {"kind": "applied", "k": 1, "version": 1, "applied_at": 100},
            {"kind": "end", "update": 1872, "version": 1},
        ],
    )
    report = OpalRunReport(tmp_path).build()
    assert report["status"] == "changes_applied"
    assert report["checkpoint_precedes_first_change"] is False
    assert report["final_program_version"] == 1
    assert report["classification"]["macro_f1"] == 0.4


def test_interrupted_run_is_not_complete(tmp_path):
    _write_run(
        tmp_path,
        [
            {"kind": "start", "backend": "codex"},
            {"kind": "interrupted", "update": 100, "version": 0},
        ],
    )
    assert OpalRunReport(tmp_path).build()["status"] == "interrupted"


def test_notebook_displays_not_started_rows_for_missing_retries(tmp_path):
    from unittest.mock import Mock

    import yaml

    nbformat = pytest.importorskip("nbformat")
    source = Path(__file__).resolve().parents[1] / "notebooks/10_opal_pilot.ipynb"
    notebook = nbformat.read(source, as_version=4)
    cell = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "records = []" in cell.source
    )
    configs = tmp_path / "configs"
    configs.mkdir()
    for prefix in ("gru-opal", "gru-opal-retry"):
        for kind in ("control", "keep", "codex"):
            recipe = f"{prefix}-{kind}"
            (configs / f"{recipe}.yaml").write_text(
                yaml.safe_dump(
                    {
                        "output_dir": f"../artifacts/{recipe}",
                    }
                )
            )
    namespace = {"ROOT": tmp_path, "logger": Mock(), "display": Mock(), "json": json}
    exec(cell.source, namespace)
    records = namespace["records"]
    assert len(records) == 6
    assert all(record["status"] == "not_started" for record in records)
    assert sum(record["group"] == "Retry" for record in records) == 3
    assert not (tmp_path / "artifacts").exists()
