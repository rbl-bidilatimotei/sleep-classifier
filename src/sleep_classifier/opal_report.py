import argparse
import json
import logging
from pathlib import Path

from sleep_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


class OpalRunReport:
    """Describe saved optimizer decisions without loading programs or credentials."""

    def __init__(self, run_dir: Path) -> None:
        """Initialize the OpalRunReport with the directory containing the run data.

        Args:
            run_dir (Path): Directory containing the run data.
        """
        self.run_dir = run_dir
        self.opal_dir = run_dir / "opal"


    @staticmethod
    def _read_object(path: Path) -> dict[str, object]:
        """Read a JSON object with an actionable corruption error.

        Args:
            path (Path): Saved metadata file.

        Returns:
            dict[str, object]: Parsed JSON object from the file.
        """
        try:
            value = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"Invalid JSON in {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}")

        return value


    def _read_events(self) -> list[dict[str, object]]:
        """Read events and reject malformed or truncated lines.

        Returns:
            list[dict[str, object]]: List of parsed event objects.
        """
        path = self.opal_dir / "events.jsonl"
        if not path.exists():
            return []
        events = []
        for index, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {index}") from error
            if not isinstance(event, dict) or not isinstance(event.get("kind"), str):
                raise ValueError(f"Invalid event in {path} at line {index}")
            events.append(event)
        return events


    def _read_decisions(self) -> list[dict[str, object]]:
        """Select operational fields while excluding prompts and raw transcripts.

        Returns:
            list[dict[str, object]]: List of filtered decision responses.
        """
        decisions = []
        fields = ("k", "requested_at", "seconds", "returncode", "backend_error", "errors", "usage")
        for path in sorted((self.opal_dir / "decisions").glob("*/response.json")):
            response = self._read_object(path)
            decisions.append({key: response.get(key) for key in fields})
        return decisions


    @staticmethod
    def _usage(end: dict[str, object], decisions: list[dict[str, object]]) -> dict[str, object]:
        """Prefer end-of-run usage and avoid double-counting response records.

        Args:
            end (dict[str, object]): Last end or interruption event.
            decisions (list[dict[str, object]]): Filtered decision responses.

        Returns:
            dict[str, object]: Aggregated usage information with source and token counts.
        """
        usage = end.get("usage")
        source = "end_event"
        if not isinstance(usage, dict):
            records = [
                decision["usage"]
                for decision in decisions
                if isinstance(decision.get("usage"), dict)
            ]
            usage = (
                {
                    key: sum(record.get(key, 0) for record in records)
                    for key in (
                        "input_tokens",
                        "cached_input_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                    )
                }
                if records
                else None
            )
            source = "decision_responses" if records else "unavailable"
        if usage is None:
            return {"source": source, "input_tokens": None, "output_tokens": None}
        inputs = usage.get("input_tokens")
        cached = usage.get("cached_input_tokens")
        return {
            "source": source,
            "input_tokens": inputs,
            "cached_input_tokens": cached,
            "uncached_input_tokens": inputs - cached
            if inputs is not None and cached is not None
            else None,
            "output_tokens": usage.get("output_tokens"),
            "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
            "total_tokens": inputs + usage["output_tokens"]
            if inputs is not None and usage.get("output_tokens") is not None
            else None,
            "note": "Cached input and reasoning output are subsets, not extra tokens",
        }


    @staticmethod
    def _status(
        backend: str | None,
        events: list[dict[str, object]],
        end: dict[str, object]
    ) -> str:
        """Classify operational success separately from classification accuracy.

        Args:
            backend (str | None): Recorded optimizer backend.
            events (list[dict[str, object]]): Validated run events.
            end (dict[str, object]): End or interruption event, if present.

        Returns:
            str: Status classification of the operational run.
        """
        if not events:
            return f"plain_{backend}" if backend in ("adam", "adamw") else "unknown_missing_events"
        if not end:
            return "incomplete"
        if end["kind"] != "end":
            return "interrupted"
        if backend == "keep":
            return "offline_keep_control"

        applied = any(event["kind"] == "applied" for event in events)
        failed = any(event["kind"] in ("rejected", "load_failed", "crashed") for event in events)
        if not applied:
            return "failed_no_changes_applied" if failed else "no_changes_applied"
        if any(event["kind"] == "reverted" for event in events):
            return "changes_applied_with_reversions"

        return "changes_applied"


    def build(self) -> dict[str, object]:
        """Build a read-only summary from available metrics and OPAL records.

        Returns:
            dict[str, object]: Summary of the run, including status, metrics, and OPAL records.
        """
        metrics_path = self.run_dir / "metrics.json"
        metrics = self._read_object(metrics_path) if metrics_path.exists() else {}
        events = self._read_events()
        decisions = self._read_decisions()

        start = next((event for event in events if event["kind"] == "start"), {})
        end = next(
            (event for event in reversed(events) if event["kind"] in ("end", "interrupted")), {}
        )

        agent = metrics.get("optimizer", {}).get("agent", {})
        backend = start.get("backend", agent.get("backend"))
        if not backend and metrics and not metrics.get("config", {}).get("optimizer_agent"):
            configured = metrics.get("config", {}).get("optimizer") or {}
            recorded = metrics.get("optimizer", {}).get("base_optimizer")
            backend = recorded.lower() if recorded in ("Adam", "AdamW") else configured.get(
                "name", "adam",
            )

        schedule = next(
            (event["schedule"] for event in reversed(events) if "schedule" in event), None
        )
        end_update = end.get("update")

        counts = {
            kind: sum(event["kind"] == kind for event in events)
            for kind in ("requested", "applied", "kept", "rejected", "load_failed", "reverted")
        }
        applied_updates = [
            event["applied_at"]
            for event in events
            if event["kind"] == "applied" and "applied_at" in event
        ]

        result = {
            "run_dir": str(self.run_dir),
            "backend": backend,
            "configured_agent": {
                key: agent.get(key)
                for key in ("model", "effort", "max_tokens", "decisions", "schedule")
            },
            "runtime": metrics.get("optimizer", {}).get("runtime"),
            "status": self._status(backend, events, end),
            "metrics_available": bool(metrics),
            "events_available": bool(events),
            "classification": {
                key: metrics.get(key) for key in ("accuracy", "macro_f1", "best_epoch")
            },
            "decision_counts": counts if events else None,
            "scheduled_updates": schedule,
            "unreached_scheduled_updates": [update for update in schedule if update > end_update]
            if schedule is not None and end_update is not None
            else None,
            "end_update": end_update,
            "final_program_version": end.get("version"),
            "first_applied_update": min(applied_updates) if applied_updates else None,
            "agent_wait_seconds": end.get("agent_wait_seconds"),
            "cost_usd": end.get("cost_usd"),
            "unknown_cost_decisions": end.get("unknown_cost_decisions"),
            "usage": self._usage(end, decisions),
            "decisions": decisions,
        }
        cap = agent.get("max_tokens")
        total = result["usage"].get("total_tokens")
        result["soft_token_cap"] = cap
        result["soft_token_cap_exceeded"] = total > cap if total is not None and cap else None
        history = metrics.get("history", [])
        updates = metrics.get("optimizer", {}).get("updates")

        if history and isinstance(updates, int) and updates % len(history) == 0:
            checkpoint_update = metrics["best_epoch"] * (updates // len(history))
            result["selected_checkpoint_update"] = checkpoint_update
            result["checkpoint_precedes_first_change"] = (
                checkpoint_update < min(applied_updates) if applied_updates else None
            )

        return result


def main() -> None:
    """Inspect saved run metadata without training or executing optimizer programs."""
    parser = argparse.ArgumentParser(description="Inspect saved OPAL run status")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    configure_logging()
    report = OpalRunReport(args.run_dir).build()
    content = json.dumps(report, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n")
    LOGGER.info("OPAL run report:\n%s", content)


if __name__ == "__main__":
    main()
