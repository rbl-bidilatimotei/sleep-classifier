import copy

import pytest

from sleep_classifier.session_protocol import COVERAGE_RULE, validate_event, validate_output


def start_event() -> dict[str, object]:
    """Return a valid zero-origin start record."""
    return {"schema_version": "1.0.0", "type": "start", "session_id": "demo",
            "participant_id": "demo", "session_start_seconds": 0}


@pytest.mark.parametrize("field,value", [
    ("label", 1), ("participant_id", "demo\n"), ("schema_version", "2.0.0"),
    ("session_start_seconds", 30),
    ("session_start_seconds", True), ("session_start_seconds", float("nan")),
])
def test_start_contract_rejects_unsupported_values(field: str, value: object) -> None:
    """Reject unsupported clocks, labels, versions, and numeric types.

    Args:
        field (str): Field to replace or add.
        value (object): Unsupported value.
    """
    event = start_event()
    event[field] = value
    with pytest.raises(ValueError):
        validate_event(event)


@pytest.mark.parametrize("time", [10**400, float("inf"), True, -1])
def test_sensor_time_requires_finite_supported_number(time: object) -> None:
    """Reject unsupported sensor times before mutating the adapter.

    Args:
        time (object): Invalid clock value.
    """
    with pytest.raises(ValueError):
        validate_event({"schema_version": "1.0.0", "type": "heart_rate",
                        "time": time, "heart_rate": 60})


def prediction_record() -> dict[str, object]:
    """Return one complete prediction response."""
    return {"schema_version": "1.0.0", "type": "prediction", "session_id": "demo",
            "participant_id": "demo", "bundle_id": "demo", "artifact_role": "evaluation_fold",
            "time": 0, "available_at": 30, "watermark_at_emission": 30, "segment_id": 0,
            "prediction": 1, "stage": "light", "probabilities": [.1, .7, .1, .1],
            "history_epochs": 1, "reset_reason": "participant_change"}


@pytest.mark.parametrize("field,value", [
    ("stage", "REM"), ("prediction", 1.0), ("prediction", 2), ("probabilities", [.1, .6, .1, .1]),
    ("available_at", 29), ("watermark_at_emission", 29),
])
def test_prediction_semantics_are_checked(field: str, value: object) -> None:
    """Reject responses that disagree with completion or class semantics.

    Args:
        field (str): Field to change.
        value (object): Inconsistent value.
    """
    record = prediction_record()
    record[field] = value
    with pytest.raises(ValueError):
        validate_output(record)


def test_valid_records_and_immutable_coverage() -> None:
    """Accept valid records and preserve the fixed coverage profile."""
    validate_event(start_event())
    validate_output(copy.deepcopy(prediction_record()))
    with pytest.raises(TypeError):
        COVERAGE_RULE["max_heart_gap_seconds"] = 20  # type: ignore[index]
