import math
from numbers import Integral
from typing import Any

from sleep_classifier.bundle_loader import BundlePredictor, EvaluationBundle
from sleep_classifier.raw_event_adapter import RawEventAdapter
from sleep_classifier.session_protocol import (
    COVERAGE_RULE,
    SCHEMA_VERSION,
    validate_event,
    validate_output,
)

_EPOCH_SECONDS = 30.0
_MOTION_GRID_FEATURE = "motion_grid_change_mean"
_COVERAGE_FIELDS = (
    "heart_sample_count",
    "heart_occupied_bin_fraction",
    "heart_largest_observation_gap",
    "motion_sample_count",
    "motion_occupied_bin_fraction",
    "motion_largest_observation_gap",
)


class RawSessionRunner:
    """Combine bounded raw events and a frozen predictor for one session."""

    def __init__(
        self, bundle: EvaluationBundle, max_buffered_events: int = 8192,
        max_epochs_per_event: int = 128,
    ) -> None:
        """Set the verified bundle and per-session resource limits.

        Args:
            bundle (EvaluationBundle): Verified evaluation bundle for frozen inference.
            max_buffered_events (int): Maximum pending unique observations across both sensors.
            max_epochs_per_event (int): Epoch outputs allowed per input. Excludes the final summary.
        """
        for name, value in (
            ("max_buffered_events", max_buffered_events),
            ("max_epochs_per_event", max_epochs_per_event),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self._bundle = bundle
        self._manifest = bundle.manifest
        self._class_names = tuple(self._manifest["model"]["class_names"])
        self._raw_features = tuple(self._manifest["model"]["raw_features"])
        self._max_buffered_events = int(max_buffered_events)
        self._max_epochs_per_event = int(max_epochs_per_event)
        self._phase = "new"
        self._adapter: RawEventAdapter | None = None
        self._predictor: BundlePredictor | None = None
        self._initial_state_sha256: str | None = None
        self._prediction_count = 0
        self._skipped_count = 0
        self._identity = {
            "schema_version": SCHEMA_VERSION,
            "bundle_id": self._manifest["bundle_id"],
            "artifact_role": self._manifest["artifact_role"],
        }


    def handle(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """Accept one event and return the records it makes available.

        Args:
            event (dict[str, Any]): Decoded version-one raw session record.

        Returns:
            list[dict[str, Any]]: Records emitted in session order.
        """
        validate_event(event)
        if self._phase in {"finished", "failed"}:
            raise ValueError(f"Cannot accept input after the session is {self._phase}")

        kind = event["type"]
        if kind == "start":
            if self._phase != "new":
                raise ValueError("Session start is allowed only once")
            return [self._start(event)]
        adapter = self._adapter
        if adapter is None:
            raise ValueError("A start event is required before session input")

        if kind == "heart_rate":
            adapter.add_heart_rate(event["time"], event["heart_rate"])
        elif kind == "motion":
            adapter.add_motion(event["time"], event["x"], event["y"], event["z"])
        elif kind == "watermark":
            return self._advance_watermark(adapter, event["sensor"], event["time"])
        elif kind == "finish":
            return self._finish(adapter, event["session_end_seconds"])
        return []


    def ensure_finished(self) -> None:
        """Require a successful finish event at the end of input."""
        if self._phase != "finished":
            raise ValueError(f"Input ended before a successful finish event (phase: {self._phase})")


    def status(self) -> dict[str, Any]:
        """Return the current phase, adapter status, and output counts."""
        return {
            "phase": self._phase,
            "adapter": self._adapter.status() if self._adapter is not None else None,
            "predicted_epochs": self._prediction_count,
            "skipped_epochs": self._skipped_count,
        }


    def _start(self, event: dict[str, Any]) -> dict[str, Any]:
        """Restore fresh inference state for the declared session.

        Args:
            event (dict[str, Any]): Validated start event.
        """
        self._identity.update(
            session_id=event["session_id"], participant_id=event["participant_id"],
        )
        try:
            adapter = RawEventAdapter(
                event["participant_id"], dict(COVERAGE_RULE),
                session_start_seconds=event["session_start_seconds"],
                max_buffered_events=self._max_buffered_events,
            )
            predictor = self._bundle.predictor(
                device=self._manifest["runtime"]["validated_device"],
            )
            integrity = predictor.integrity()
            state_sha256 = integrity["state_sha256"]
            if not isinstance(state_sha256, str):
                raise ValueError("Predictor must report a model state digest")
            if not integrity["evaluation_mode"] or integrity["gradients_created"]:
                raise ValueError("Predictor must start in evaluation mode without gradients")
            record = {
                **self._identity,
                "type": "session_started",
                "checkpoint_sha256": self._manifest["files"]["checkpoint"]["sha256"],
                "coverage_profile": "moderate-v1",
                "class_names": list(self._class_names),
            }
            validate_output(record)
        except Exception:
            self._phase = "failed"
            raise

        self._adapter = adapter
        self._predictor = predictor
        self._initial_state_sha256 = state_sha256
        self._phase = "active"
        return record


    def _advance_watermark(
        self, adapter: RawEventAdapter, sensor: str, time: float,
    ) -> list[dict[str, Any]]:
        """Check the output limit before advancing a sensor watermark.

        Args:
            adapter (RawEventAdapter): Active sensor adapter.
            sensor (str): Sensor whose watermark advances.
            time (float): Proposed watermark in session seconds.
        """
        status = adapter.status()
        proposed_watermarks = status["watermarks"]
        proposed_watermarks[sensor] = float(time)
        self._check_epoch_limit(status["next_epoch_start"], min(proposed_watermarks.values()))
        adapter.advance_watermark(sensor, time)
        return self._drain(adapter)


    def _check_epoch_limit(self, next_epoch_start: float, effective_watermark: float) -> None:
        """Reject a proposed progress event that would exceed its epoch output limit.

        Args:
            next_epoch_start (float): Start time of the next pending epoch.
            effective_watermark (float): Proposed lowest sensor watermark.
        """
        completed = math.floor((effective_watermark - next_epoch_start) / _EPOCH_SECONDS)
        if completed > self._max_epochs_per_event:
            raise ValueError("Progress event would exceed max_epochs_per_event")


    def _finish(self, adapter: RawEventAdapter, end: float) -> list[dict[str, Any]]:
        """Close the input and report the final counts and model integrity.

        Args:
            adapter (RawEventAdapter): Active sensor adapter.
            end (float): Exclusive session end in seconds.
        """
        self._check_epoch_limit(adapter.status()["next_epoch_start"], float(end))
        adapter.finish(end)
        records = self._drain(adapter)
        try:
            summary = self._final_summary(adapter)
            validate_output(summary)
        except Exception:
            self._phase = "failed"
            raise

        self._phase = "finished"
        records.append(summary)
        return records


    def _drain(self, adapter: RawEventAdapter) -> list[dict[str, Any]]:
        """Drain completed epochs and validate each output before counting it.

        Args:
            adapter (RawEventAdapter): Sensor adapter with accepted watermarks.
        """
        records: list[dict[str, Any]] = []
        try:
            while (row := adapter.pop_completed()) is not None:
                record = self._epoch_record(row)
                validate_output(record)
                records.append(record)
                self._prediction_count += int(row["eligible"])
                self._skipped_count += int(not row["eligible"])
        except Exception:
            # A popped epoch or predictor update cannot be retried
            self._phase = "failed"
            raise
        return records


    def _epoch_record(self, row: dict[str, Any]) -> dict[str, Any]:
        """Build a prediction or a skipped-epoch record from sensor features.

        Args:
            row (dict[str, Any]): Completed sensor epoch with coverage and features.
        """
        common = {
            **self._identity,
            "time": row["time"],
            "available_at": row["available_at"],
            "watermark_at_emission": row["watermark_at_emission"],
            "segment_id": row["segment_id"],
        }
        if not row["eligible"]:
            return {
                **common,
                "type": "skipped_epoch",
                "reason": "insufficient_sensor_coverage",
                "coverage": {field: row[field] for field in _COVERAGE_FIELDS},
            }

        # Sensor segments let the predictor apply its normal reset on the next eligible epoch
        response = self._predict_epoch(row)
        prediction = response["prediction"]
        return {
            **common,
            "type": "prediction",
            "prediction": prediction,
            "stage": self._class_names[prediction],
            "probabilities": response["probabilities"],
            "history_epochs": response["history_epochs"],
            "reset_reason": response["reset_reason"],
        }


    def _predict_epoch(self, row: dict[str, Any]) -> dict[str, Any]:
        """Predict an eligible epoch using the saved motion imputation when needed.

        Args:
            row (dict[str, Any]): Completed epoch with eligible sensor coverage.
        """
        features = {name: float(row[name]) for name in self._raw_features}
        if math.isnan(features[_MOTION_GRID_FEATURE]):
            features[_MOTION_GRID_FEATURE] = self._bundle.motion_imputation_mean
        for name, value in features.items():
            if not math.isfinite(value):
                raise ValueError(f"Eligible epoch has a nonfinite feature: {name}")

        predictor = self._predictor
        assert predictor is not None
        return predictor.predict({
            "participant_id": row["participant_id"],
            "segment_id": row["segment_id"],
            "time": row["time"],
            "features": features,
        })


    def _final_summary(self, adapter: RawEventAdapter) -> dict[str, Any]:
        """Verify frozen model state and collect final counters.

        Args:
            adapter (RawEventAdapter): Adapter drained through the session end.
        """
        predictor = self._predictor
        assert predictor is not None
        integrity = predictor.integrity()
        unchanged = integrity["state_sha256"] == self._initial_state_sha256
        if not unchanged or not integrity["evaluation_mode"] or integrity["gradients_created"]:
            raise ValueError("Predictor integrity check failed at session finish")

        status = adapter.status()
        return {
            **self._identity,
            "type": "session_finished",
            "emitted_epochs": status["emitted_epochs"],
            "predicted_epochs": self._prediction_count,
            "skipped_epochs": self._skipped_count,
            "duplicate_events": status["duplicate_events"],
            "discarded_tail_events": status["discarded_tail_events"],
            "peak_buffered_events": status["peak_buffered_events"],
            "max_buffered_events": self._max_buffered_events,
            "integrity": {**integrity, "weights_unchanged": unchanged},
        }
