import math
from functools import cache
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from jsonschema import Draft202012Validator, ValidationError

from sleep_classifier.bundle_validation import _validate_json_values, strict_json_loads

SCHEMA_VERSION = "1.0.0"
COVERAGE_RULE: Mapping[str, float] = MappingProxyType({
    "min_heart_readings": 3.0, "max_heart_gap_seconds": 15.0,
    "min_motion_bin_fraction": 0.8, "max_motion_gap_seconds": 5.0,
})


@cache
def _schema(name: str) -> dict[str, Any]:
    """Load a local protocol schema from the installed wheel or source checkout.

    Args:
        name (str): Known event or output schema name.
    """
    if name not in {"event", "output"}:
        raise ValueError("Unknown raw session schema")
    resource = files("sleep_classifier").joinpath("session_schemas", "v1", f"{name}.schema.json")
    if not resource.is_file():
        resource = (Path(__file__).resolve().parents[2]
                    / "schemas/raw-session/v1" / f"{name}.schema.json")
    schema = strict_json_loads(resource.read_bytes())
    Draft202012Validator.check_schema(schema)
    return schema


@cache
def _validator(name: str, kind: str) -> Draft202012Validator:
    """Cache the explicit record variant to avoid scanning unrelated variants per sensor event.

    Args:
        name (str): Event or output schema name.
        kind (str): Declared record type inside that schema.
    """
    definitions = _schema(name)["$defs"]
    if kind not in definitions:
        raise ValueError(f"Unknown raw session {name} type")
    return Draft202012Validator(definitions[kind])


def _validate(record: dict[str, Any], name: str) -> None:
    """Check exact JSON types, finite numbers, and the selected versioned record shape.

    Args:
        record (dict[str, Any]): Decoded protocol record.
        name (str): Event or output schema name.
    """
    _validate_json_values(record)
    if type(record) is not dict or not isinstance(record.get("type"), str):
        raise ValueError("Raw session record requires an object and a string type")
    for value in record.values():
        if type(value) in (int, float):
            try:
                finite = math.isfinite(float(value))
            except OverflowError as error:
                raise ValueError("Raw session number exceeds the supported range") from error
            if not finite:
                raise ValueError("Raw session numbers must be finite")
    try:
        _validator(name, record["type"]).validate(record)
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path)
        raise ValueError(f"Invalid raw session {name} field {location or 'record'}") from error


def validate_event(record: dict[str, Any]) -> None:
    """Validate a label-free input event without changing session state.

    Args:
        record (dict[str, Any]): One decoded raw session event.
    """
    _validate(record, "event")


def validate_output(record: dict[str, Any]) -> None:
    """Validate a finite prediction, skip, or lifecycle response before serialization.

    Args:
        record (dict[str, Any]): One completed protocol response.
    """
    _validate(record, "output")
    if record["type"] in {"prediction", "skipped_epoch"}:
        if (abs(record["available_at"]-record["time"]-30) > 1e-8
                or record["watermark_at_emission"] < record["available_at"]):
            raise ValueError("Raw session output violates epoch completion timing")
    if record["type"] == "prediction":
        if type(record["prediction"]) is not int:
            raise ValueError("Raw session prediction code must be an integer")
        probabilities = record["probabilities"]
        if abs(sum(probabilities)-1) > 1e-6:
            raise ValueError("Raw session probabilities must sum to one")
        if record["prediction"] != probabilities.index(max(probabilities)):
            raise ValueError("Raw session class must match its probability argmax")
        if record["stage"] != ("wake", "light", "deep", "REM")[record["prediction"]]:
            raise ValueError("Raw session stage must match its class code")
    if record["type"] == "session_finished":
        if record["emitted_epochs"] != record["predicted_epochs"] + record["skipped_epochs"]:
            raise ValueError("Session epoch counters do not balance")
        if record["peak_buffered_events"] > record["max_buffered_events"]:
            raise ValueError("Session peak exceeds its buffer limit")
