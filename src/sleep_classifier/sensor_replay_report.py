import re
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from sleep_classifier.baseline import CLASS_LABELS

EPOCH_KEYS = ["participant_id", "time"]
PROBABILITY_COLUMNS = [f"probability_{label}" for label in CLASS_LABELS]
PARTICIPANT_ID = re.compile(r"[A-Za-z0-9_-]+")
NLL_FLOOR = 1e-12


class SensorReplayReport:
    """Score completed sensor predictions against saved validation evidence."""

    @staticmethod
    def build(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
        """Join predictions and labels by participant and exact epoch start time.

        Args:
            candidate (pd.DataFrame): All sensor replay predictions, without targets.
            reference (pd.DataFrame): Saved predictions and targets for eligible epochs.

        Returns:
            dict[str, Any]: Coverage, paired scores, and scores for all reference epochs.
        """
        candidate_rows = SensorReplayReport._validated(candidate, "candidate")
        reference_rows = SensorReplayReport._validated(reference, "reference")

        shared_keys = reference_rows.index.intersection(candidate_rows.index)
        reference_only = reference_rows.index.difference(shared_keys)
        shared_candidate = candidate_rows.loc[shared_keys]
        shared_reference = reference_rows.loc[shared_keys]
        return {
            "scopes": {
                "candidate_epochs": len(candidate_rows),
                "reference_epochs": len(reference_rows),
                "shared_scored_epochs": len(shared_keys),
                "candidate_only_epochs": len(candidate_rows) - len(shared_keys),
                "reference_only_epochs": len(reference_only),
            },
            "reference_only_identities": reference_only.to_frame(index=False).to_dict("records"),
            "reference": SensorReplayReport._metrics(shared_reference, shared_reference.target),
            "candidate": SensorReplayReport._metrics(shared_candidate, shared_reference.target),
            "reference_all_eligible": SensorReplayReport._metrics(
                reference_rows, reference_rows.target
            ),
            "changes": SensorReplayReport._changes(shared_candidate, shared_reference),
            "participants": {
                participant: {
                    "shared_scored_epochs": len(rows),
                    "reference": SensorReplayReport._metrics(rows, rows.target),
                    "candidate": SensorReplayReport._metrics(
                        shared_candidate.loc[rows.index], rows.target
                    ),
                }
                for participant, rows in shared_reference.groupby(level="participant_id")
            },
            "validation_nll_definition": (
                "Mean negative log probability of the true class with no label smoothing. "
                f"Probabilities below {NLL_FLOOR:g} are clipped to {NLL_FLOOR:g}."
            ),
        }

    @staticmethod
    def _validated(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        """Validate the prediction table and index it by epoch identity.

        Args:
            frame (pd.DataFrame): Input predictions and optional history counts.
            source (str): Candidate or reference table name.

        Raises:
            ValueError: If the input violates the prediction table contract.
        """
        columns = [*EPOCH_KEYS, "prediction", *PROBABILITY_COLUMNS]
        class_columns = ["prediction"]
        if source == "reference":
            columns.append("target")
            class_columns.append("target")
        elif "target" in frame:
            raise ValueError("The candidate table must not contain a target column")
        if not frame.columns.is_unique or not set(columns).issubset(frame.columns):
            raise ValueError(f"The {source} table requires unique columns including {columns}")
        if "history_epochs" in frame:
            columns.append("history_epochs")

        rows = frame[columns].copy()
        if any(not isinstance(value, str) or PARTICIPANT_ID.fullmatch(value) is None
               for value in rows.participant_id):
            raise ValueError(f"The {source} table contains an invalid participant_id")

        for column in columns[1:]:
            if any(not isinstance(value, Real) or isinstance(value, bool)
                   for value in rows[column]):
                raise ValueError(f"The {source} table requires real values for {column}")
            values = rows[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"The {source} table requires finite values for {column}")
            rows[column] = values

        if rows.duplicated(EPOCH_KEYS).any():
            raise ValueError(f"The {source} table contains duplicate epoch identities")
        for column in class_columns:
            if not rows[column].isin(CLASS_LABELS).all():
                raise ValueError(
                    f"The {source} table requires class codes 0 through 3 for {column}"
                )
            rows[column] = rows[column].astype(int)
        if "history_epochs" in rows:
            history = rows.history_epochs.to_numpy()
            if (history < 1).any() or not np.array_equal(history, np.floor(history)):
                raise ValueError(f"The {source} table requires positive integer history_epochs")

        probabilities = rows[PROBABILITY_COLUMNS].to_numpy()
        if ((probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
            raise ValueError(f"The {source} probabilities must lie in [0, 1] and sum to 1")
        if not np.array_equal(rows.prediction, probabilities.argmax(axis=1)):
            raise ValueError(f"The {source} predictions must match the probability argmax")
        return rows.set_index(EPOCH_KEYS)

    @staticmethod
    def _metrics(rows: pd.DataFrame, target: pd.Series) -> dict[str, Any] | None:
        """Calculate four-class scores for one population.

        Args:
            rows (pd.DataFrame): Scored predictions in target order.
            target (pd.Series): True class codes for these predictions.
        """
        if rows.empty:
            return None
        precision, recall, f1, support = precision_recall_fscore_support(
            target, rows.prediction, labels=CLASS_LABELS, zero_division=0
        )
        probabilities = rows[PROBABILITY_COLUMNS].to_numpy()
        true_probabilities = probabilities[np.arange(len(rows)), target.to_numpy()]
        return {
            "accuracy": float(accuracy_score(target, rows.prediction)),
            "macro_f1": float(f1.mean()),
            "validation_nll": float(-np.log(true_probabilities.clip(min=NLL_FLOOR)).mean()),
            "classes": {
                str(label): {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(CLASS_LABELS)
            },
            "confusion_matrix": confusion_matrix(
                target, rows.prediction, labels=CLASS_LABELS
            ).tolist(),
        }

    @staticmethod
    def _changes(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, int | float | None]:
        """Summarize prediction changes on the shared epoch identities.

        Args:
            candidate (pd.DataFrame): Sensor predictions for the paired epochs.
            reference (pd.DataFrame): Saved predictions for the paired epochs.
        """
        differences = candidate[PROBABILITY_COLUMNS] - reference[PROBABILITY_COLUMNS]
        changes = {
            "class_disagreements": int((candidate.prediction != reference.prediction).sum()),
            "max_abs_probability_difference": (
                None if differences.empty else float(np.abs(differences.to_numpy()).max())
            ),
        }
        if "history_epochs" in candidate and "history_epochs" in reference:
            changes["history_disagreements"] = int(
                (candidate.history_epochs != reference.history_epochs).sum()
            )
        return changes
