from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

STAGE_NAMES = ("Wake", "Light", "Deep", "REM")


class AuditVisualization:
    """Render development audit evidence without fitting or changing a model."""

    def __init__(self, rows: pd.DataFrame, summary: dict[str, Any]) -> None:
        """Store unique target rows and their recorded audit summary.

        Args:
            rows (pd.DataFrame): Aligned development predictions and causal audit features.
            summary (dict[str, Any]): Audit provenance and descriptive results.
        """
        required = {"participant_id", "time", "target", "segment_id", "prediction_reference",
                    "persistent_to_light", "reference_correct"}
        if not required.issubset(rows.columns) or rows.empty:
            raise ValueError("Audit visualization requires nonempty aligned prediction rows")
        self.rows = rows.copy()
        if self.rows[list(required)].isna().any().any():
            raise ValueError("Target identities and prediction flags must be complete")
        self.rows["participant_id"] = self.rows.participant_id.astype(str)
        self.rows["time"] = pd.to_numeric(self.rows.time).astype(float)
        for column in ("persistent_to_light", "reference_correct"):
            if not self.rows[column].isin((True, False)).all():
                raise ValueError("Audit prediction flags must be boolean values")
        if self.rows.duplicated(["participant_id", "time"]).any():
            raise ValueError("Audit targets must be unique within each participant")
        if not np.isfinite(self.rows.time.to_numpy(dtype=float)).all():
            raise ValueError("Audit target times must be finite")
        for column in ["target", *self._prediction_columns()]:
            if not self.rows[column].isin(range(4)).all():
                raise ValueError("Audit stage codes must be integers from zero through three")
        self.rows = self.rows.sort_values(["participant_id", "time"], kind="stable")
        self.summary = dict(summary)


    def _prediction_columns(self) -> list[str]:
        """Return selected-checkpoint prediction fields in their input order."""
        return [column for column in self.rows if column.startswith("prediction_")]


    def _payload(self) -> str:
        """Encode finite JSON values and escape script-closing text safely."""
        # DataFrame JSON encodes missing optional feature values as null
        rows = json.loads(self.rows.to_json(orient="records", double_precision=12))
        payload = json.dumps({"rows": rows, "summary": self.summary,
                              "recipes": self._prediction_columns()}, allow_nan=False,
                             default=self._json_value)
        for character, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                   ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
            payload = payload.replace(character, escaped)
        return payload


    @staticmethod
    def _json_value(value: Any) -> Any:
        """Convert NumPy scalar metadata while rejecting unsupported objects.

        Args:
            value (Any): Nonstandard scalar found in the audit summary.
        """
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported audit metadata type: {type(value).__name__}")


    def write_html(self, output_path: Path) -> Path:
        """Write a portable local viewer with all data and scripts embedded.

        Args:
            output_path (Path): Standalone HTML destination.
        """
        template = Path(__file__).with_name("templates") / "causal_audit.html"
        document = template.read_text().replace("__AUDIT_PAYLOAD__", self._payload())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
        return output_path


    def plot_distributions(self) -> Figure:
        """Plot participant-stage medians without treating epochs as independent people."""
        import matplotlib.pyplot as plt

        features = [
            ("heart_rate_mean", "Epoch heart rate", "Heart rate"),
            ("heart_rate_delta_past", "Heart rate minus past mean", "Heart-rate difference"),
            ("motion_magnitude_std", "Epoch motion-magnitude variation", "Acceleration units"),
            ("motion_vector_change_mean", "Within-epoch axis-vector change", "Acceleration units"),
        ]
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for axis, (column, title, units) in zip(axes.flat, features, strict=True):
            self._distribution_panel(axis, column)
            axis.set(title=title, ylabel=units, xticks=range(4), xticklabels=STAGE_NAMES)
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle("Development descriptions: one median per participant and true stage")
        return figure


    def _distribution_panel(self, axis: Any, column: str) -> None:
        """Draw available participant-stage medians and annotate their supports.

        Args:
            axis (Any): Matplotlib axis receiving one descriptive feature panel.
            column (str): Optional numeric audit feature.
        """
        if column not in self.rows:
            axis.text(0.5, 0.5, "Feature unavailable", ha="center", transform=axis.transAxes)
            return
        medians = self.rows.groupby(["participant_id", "target"])[column].median()
        values = [medians.loc[medians.index.get_level_values("target") == stage].dropna()
                  for stage in range(4)]
        positions = [stage for stage, data in enumerate(values) if len(data)]
        if positions:
            axis.boxplot([values[stage].to_numpy() for stage in positions], positions=positions,
                         widths=0.5, showfliers=False)
        supports = " / ".join(str(len(data)) for data in values)
        axis.text(0.01, 0.98, f"Participant-stage medians: {supports}", va="top",
                  fontsize=8, transform=axis.transAxes)


    def plot_persistent_errors(self) -> Figure:
        """Show persistent Deep/REM-to-Light target counts for each development participant."""
        import matplotlib.pyplot as plt

        people = sorted(self.rows.participant_id.unique())
        errors = self.rows.loc[self.rows.persistent_to_light.astype(bool)]
        table = pd.crosstab(errors.participant_id, errors.target).reindex(
            index=people, columns=[2, 3], fill_value=0,
        )
        figure, axis = plt.subplots(figsize=(11, max(4, len(people) * 0.23)),
                                    layout="constrained")
        axis.barh(people, table[2], label="Deep → Light", color="#5874b8")
        axis.barh(people, table[3], left=table[2], label="REM → Light", color="#b75b8e")
        axis.set(xlabel="Targets classified as Light by every compared recipe",
                 ylabel="Development participant", title="Persistent errors across related recipes")
        axis.legend()
        axis.grid(axis="x", alpha=0.2)
        return figure
