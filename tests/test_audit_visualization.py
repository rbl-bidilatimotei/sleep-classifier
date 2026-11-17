import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from sleep_classifier.audit_visualization import AuditVisualization  # noqa: E402


def _rows() -> pd.DataFrame:
    """Create two development participants with a gap and unavailable startup history."""
    return pd.DataFrame({
        "participant_id": ["a"] * 4 + ["b"] * 4,
        "time": [0, 30, 90, 120] * 2,
        "target": [0, 1, 2, 3] * 2,
        "segment_id": [0, 0, 1, 1] * 2,
        "fold": ["fold-01"] * 4 + ["fold-02"] * 4,
        "prediction_reference": [0, 1, 1, 1] * 2,
        "prediction_patch_size_2": [0, 1, 1, 3] * 2,
        "prediction_adamw": [0, 1, 1, 1] * 2,
        "prediction_label_smoothing": [0, 1, 1, 1] * 2,
        "persistent_to_light": [False, False, True, False] * 2,
        "reference_correct": [True, True, False, False] * 2,
        "heart_rate_mean": [60, 62, 65, 70] * 2,
        "heart_rate_delta_past": [np.nan, 2, np.nan, 5] * 2,
        "motion_magnitude_std": [0.01, 0.02, 0.01, 0.03] * 2,
        "motion_vector_change_mean": [0.02, 0.05, 0.04, np.nan] * 2,
    })


def test_html_embeds_safe_json_and_local_interactive_controls(tmp_path: Path) -> None:
    """Keep user data inert and expose offline controls without external resources.

    Args:
        tmp_path (Path): Temporary standalone report directory.
    """
    attack = "</script><script>alert('untrusted')</script>"
    destination = AuditVisualization(_rows(), {"note": attack}).write_html(tmp_path / "audit.html")
    document = destination.read_text()
    marker = '<script id="audit-data" type="application/json">'
    payload = document.split(marker, 1)[1].split("</script>", 1)[0]
    decoded = json.loads(payload)

    assert decoded["summary"]["note"] == attack
    assert attack not in document
    assert decoded["rows"][0]["heart_rate_delta_past"] is None
    assert decoded["recipes"] == [
        "prediction_reference", "prediction_patch_size_2", "prediction_adamw",
        "prediction_label_smoothing",
    ]
    assert '<script src=' not in document and '<link ' not in document
    assert 'fetch(' not in document and 'XMLHttpRequest' not in document
    for identifier in ("participant", "recipe", "window-size", "window-start", "stage-filter",
                       "error-filter", "target-table", "heart-chart", "motion-chart"):
        assert f'id="{identifier}"' in document
    assert "Timeline and feature charts remain unfiltered" in document
    assert "previous.segment_id === row.segment_id" in document
    assert "No model training runs here" in document


def test_static_plots_show_participant_stage_supports_and_error_counts() -> None:
    """Show descriptive feature supports and persistent errors without model fitting."""
    import matplotlib.pyplot as plt

    visualizer = AuditVisualization(_rows(), {})
    distributions = visualizer.plot_distributions()
    errors = visualizer.plot_persistent_errors()
    assert len(distributions.axes) == 4
    assert "one median per participant" in distributions._suptitle.get_text()
    assert "2 / 2 / 2 / 2" in distributions.axes[0].texts[0].get_text()
    assert len(errors.axes[0].patches) == 4
    assert sum(patch.get_width() for patch in errors.axes[0].patches) == 2
    plt.close(distributions)
    plt.close(errors)


def test_visualizer_handles_empty_persistent_slice_and_missing_optional_features() -> None:
    """Preserve unavailable feature values and empty errors as descriptive missingness."""
    import matplotlib.pyplot as plt

    rows = _rows().drop(columns=["heart_rate_delta_past", "motion_vector_change_mean"])
    rows["persistent_to_light"] = False
    visualizer = AuditVisualization(rows, {})
    distributions = visualizer.plot_distributions()
    errors = visualizer.plot_persistent_errors()
    assert distributions.axes[1].texts[0].get_text() == "Feature unavailable"
    assert sum(patch.get_width() for patch in errors.axes[0].patches) == 0
    plt.close(distributions)
    plt.close(errors)


@pytest.mark.parametrize("mutation", ["duplicate", "nonfinite", "stage", "missing", "flag"])
def test_visualizer_rejects_invalid_epoch_identities(mutation: str) -> None:
    """Reject row sets that cannot support a correctly aligned time view.

    Args:
        mutation (str): Invalid epoch condition.
    """
    rows = _rows()
    if mutation == "duplicate":
        rows = pd.concat([rows, rows.iloc[:1]])
    elif mutation == "nonfinite":
        rows["time"] = rows.time.astype(float)
        rows.loc[0, "time"] = float("inf")
    elif mutation == "stage":
        rows.loc[0, "prediction_reference"] = 5
    elif mutation == "flag":
        rows["persistent_to_light"] = "no"
    else:
        rows = rows.drop(columns="target")
    with pytest.raises(ValueError):
        AuditVisualization(rows, {})


def test_dashboard_javascript_controls_preserve_time_and_missingness(tmp_path: Path) -> None:
    """Exercise render functions offline without launching a browser or making network requests.

    Args:
        tmp_path (Path): Temporary dashboard output directory.
    """
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional for offline JavaScript contract checks")
    output = AuditVisualization(_rows(), {}).write_html(tmp_path / "dashboard.html")
    harness = Path(__file__).parent / "fixtures/audit_dashboard_contract.js"
    completed = subprocess.run(
        [node, str(harness), str(output)], capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert "contracts passed" in completed.stdout
