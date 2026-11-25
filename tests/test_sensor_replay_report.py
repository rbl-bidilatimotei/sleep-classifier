import json
from math import log
from typing import Any

import pandas as pd
import pytest

from sleep_classifier.sensor_replay_report import SensorReplayReport


def _row(
    participant: str,
    time: float,
    probabilities: tuple[float, float, float, float],
    history: int = 1,
) -> dict[str, Any]:
    """Build a sensor prediction for one synthetic epoch.

    Args:
        participant (str): Participant identifier.
        time (float): Session-relative epoch start.
        probabilities (tuple[float, float, float, float]): Class probabilities in code order.
        history (int): Number of available history epochs.
    """
    return {
        "participant_id": participant,
        "time": time,
        "prediction": probabilities.index(max(probabilities)),
        "history_epochs": history,
        **{f"probability_{index}": value for index, value in enumerate(probabilities)},
    }


def test_scores_only_exact_shared_identities_and_discloses_missing_epochs() -> None:
    """Compare identical targets by epoch while disclosing both unmatched populations."""
    reference = pd.DataFrame([
        _row("subject-A", 0.0, (0.8, 0.2, 0.0, 0.0)),
        _row("subject-A", 30.0, (0.1, 0.9, 0.0, 0.0), history=2),
        _row("subject-B", 0.0, (0.1, 0.1, 0.7, 0.1)),
        _row("subject-B", 30.0, (0.8, 0.0, 0.1, 0.1)),
    ]).assign(target=[0, 1, 2, 3], fold=["fold-a", "fold-a", "fold-b", "fold-b"])
    candidate = pd.DataFrame([
        _row("subject-B", 0.0, (0.1, 0.1, 0.7, 0.1)),
        _row("subject-A", 60.0, (0.1, 0.0, 0.1, 0.8)),
        _row("subject-A", 30.0, (0.8, 0.2, 0.0, 0.0)),
        _row("subject-A", 0.0, (0.75, 0.25, 0.0, 0.0)),
    ]).assign(fold=["fold-b", "fold-a", "fold-a", "fold-a"])

    report = SensorReplayReport.build(candidate, reference)

    assert report["scopes"] == {
        "candidate_epochs": 4, "reference_epochs": 4, "shared_scored_epochs": 3,
        "candidate_only_epochs": 1, "reference_only_epochs": 1,
    }
    assert report["reference_only_identities"] == [{"participant_id": "subject-B", "time": 30.0}]
    assert report["reference"]["accuracy"] == 1.0
    assert report["reference"]["macro_f1"] == 0.75
    assert report["reference_all_eligible"]["accuracy"] == 0.75
    assert report["reference_all_eligible"]["classes"]["3"]["support"] == 1
    assert report["candidate"]["accuracy"] == pytest.approx(2 / 3)
    assert report["candidate"]["macro_f1"] == pytest.approx((2 / 3 + 1) / 4)
    assert report["candidate"]["validation_nll"] == pytest.approx(
        -(log(0.75) + log(0.2) + log(0.7)) / 3
    )
    assert report["candidate"]["classes"]["0"] == {
        "precision": 0.5, "recall": 1.0, "f1": pytest.approx(2 / 3), "support": 1
    }
    assert report["candidate"]["classes"]["3"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0
    }
    assert report["candidate"]["confusion_matrix"] == [
        [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]
    ]
    assert report["changes"] == {
        "class_disagreements": 1,
        "max_abs_probability_difference": pytest.approx(0.7),
        "history_disagreements": 1,
    }
    participant_a = report["participants"]["subject-A"]
    assert participant_a["shared_scored_epochs"] == 2
    assert participant_a["candidate"]["accuracy"] == 0.5
    assert participant_a["reference"]["accuracy"] == 1.0
    assert report["participants"]["subject-B"]["shared_scored_epochs"] == 1
    json.dumps(report, allow_nan=False)


def test_hard_label_nll_clips_zero_probability() -> None:
    """Use the true class probability and omit unavailable history comparison."""
    candidate = pd.DataFrame([_row("01", 0.0, (1.0, 0.0, 0.0, 0.0))])
    reference = candidate.assign(target=1).drop(columns="history_epochs")

    report = SensorReplayReport.build(candidate, reference)

    assert report["candidate"]["validation_nll"] == pytest.approx(-log(1e-12))
    assert "history_disagreements" not in report["changes"]


@pytest.mark.parametrize("empty_candidate", [False, True])
def test_report_can_disclose_no_overlap(empty_candidate: bool) -> None:
    """Keep coverage and reference metrics when paired scores are unavailable.

    Args:
        empty_candidate (bool): Whether the candidate table contains any epochs.
    """
    candidate = pd.DataFrame([_row("01", 0.0, (1.0, 0.0, 0.0, 0.0))])
    reference = candidate.assign(time=[1e-8], target=0)
    if empty_candidate:
        candidate = candidate.head(0)

    report = SensorReplayReport.build(candidate, reference)

    assert report["scopes"]["shared_scored_epochs"] == 0
    assert report["scopes"]["candidate_only_epochs"] == len(candidate)
    assert report["reference"] is None
    assert report["candidate"] is None
    assert report["reference_all_eligible"]["accuracy"] == 1.0
    assert report["changes"]["max_abs_probability_difference"] is None
    assert report["participants"] == {}


def test_candidate_labels_are_rejected() -> None:
    """Reject candidate tables that contain a target column."""
    labeled = pd.DataFrame([_row("01", 0.0, (1.0, 0.0, 0.0, 0.0))]).assign(target=0)

    with pytest.raises(ValueError, match="candidate.*target"):
        SensorReplayReport.build(labeled, labeled)


@pytest.mark.parametrize(
    ("source", "column", "value", "error"),
    [
        ("candidate", "participant_id", "../01", "participant_id"),
        ("reference", "participant_id", 1, "participant_id"),
        ("candidate", "time", float("nan"), "finite.*time"),
        ("reference", "time", float("inf"), "finite.*time"),
        ("candidate", "time", "0.0", "real.*time"),
        ("reference", "time", True, "real.*time"),
        ("candidate", "prediction", 1, "argmax"),
        ("reference", "prediction", 3.5, "class codes"),
        ("reference", "target", 4, "class codes"),
        ("candidate", "probability_0", 0.8, "sum to 1"),
        ("candidate", "probability_0", 0.999998, "sum to 1"),
        ("reference", "probability_0", float("inf"), "finite.*probability"),
        ("reference", "probability_2", -1e-8, "lie in"),
        ("candidate", "history_epochs", 0, "positive integer"),
    ],
)
def test_rejects_invalid_prediction_evidence(
    source: str, column: str, value: object, error: str,
) -> None:
    """Reject malformed rows before the epoch identity join.

    Args:
        source (str): Table that contains an invalid value.
        column (str): Column to replace.
        value (object): Invalid value to insert.
        error (str): Expected error pattern.
    """
    candidate = pd.DataFrame([_row("01", 0.0, (1.0, 0.0, 0.0, 0.0))])
    tables = {"candidate": candidate, "reference": candidate.assign(time=30.0, target=0)}
    tables[source] = tables[source].assign(**{column: [value]})

    with pytest.raises(ValueError, match=error):
        SensorReplayReport.build(**tables)


@pytest.mark.parametrize("source", ["candidate", "reference"])
def test_rejects_duplicate_epoch_identities_across_fold_values(source: str) -> None:
    """Require one prediction for each participant and time in a pooled table.

    Args:
        source (str): Table that contains the duplicate identity.
    """
    candidate = pd.DataFrame([_row("01", 0.0, (1.0, 0.0, 0.0, 0.0))]).assign(fold="fold-a")
    tables = {"candidate": candidate, "reference": candidate.assign(target=0)}
    tables[source] = pd.concat([tables[source], tables[source].assign(fold="fold-b")])

    with pytest.raises(ValueError, match="duplicate epoch identities"):
        SensorReplayReport.build(**tables)
