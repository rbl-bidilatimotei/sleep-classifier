import json
from pathlib import Path
from typing import Any

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

    assert decoded["summary"] == {"note": attack}
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


def _extra_summary() -> dict[str, Any]:
    """Describe optional numeric views without supplying HTML or element identifiers."""
    return {
        "diagnostic_columns": ["motion_magnitude_std", "motion_vector_change_mean", "missing"],
        "feature_definitions": {"motion_vector_change_mean": "Source-unit vector differences"},
        "extra_charts": [{"title": "Source-unit motion diagnostics", "series": [
            {"key": "motion_vector_change_mean", "label": "Vector change", "color": "#a04B6b"},
            {"key": "heart_rate_delta_past", "label": "Startup missing", "color": "#176b9c"},
        ]}],
        "extra_table_columns": [
            {"key": "motion_vector_change_mean", "label": "Recorded change", "digits": 4},
            {"key": "heart_rate_delta_past", "label": "Recorded delta", "digits": 1},
        ],
    }


def test_optional_distributions_use_available_catalog_columns_and_labels() -> None:
    """Use declared numeric fields while preserving descriptive participant-stage supports."""
    import matplotlib.pyplot as plt

    figure = AuditVisualization(_rows(), _extra_summary()).plot_distributions()
    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Motion magnitude std"
    assert figure.axes[1].get_title() == "Source-unit vector differences"
    assert "2 / 2 / 2 / 0" in figure.axes[1].texts[0].get_text()
    plt.close(figure)


def test_optional_distributions_handle_unavailable_catalog() -> None:
    """Keep absent optional fields visible as an unavailable description."""
    import matplotlib.pyplot as plt

    figure = AuditVisualization(_rows(), {"diagnostic_columns": ["missing"]}).plot_distributions()
    assert len(figure.axes) == 1
    assert figure.axes[0].texts[0].get_text() == "Declared features unavailable"
    plt.close(figure)


@pytest.mark.parametrize("catalog", [
    {"diagnostic_columns": None},
    {"diagnostic_columns": []},
    {"diagnostic_columns": ["fold"]},
    {"diagnostic_columns": ["prediction_reference"]},
    {"diagnostic_columns": ["reference_correct"]},
    {"diagnostic_columns": ["time"]},
    {"diagnostic_columns": ["motion_vector_change_mean"] * 2},
    {"diagnostic_columns": ["motion_vector_change_mean"],
     "feature_definitions": {"motion_vector_change_mean": 2}},
    {"extra_charts": {}},
    {"extra_charts": [{"title": "Unsafe ID", "series": [], "id": "stage-chart"}]},
    {"extra_charts": [{"title": "", "series": []}]},
    {"extra_charts": [{"title": "Bad color", "series": [
        {"key": "motion_vector_change_mean", "label": "Change", "color": "url(x)"},
    ]}]},
    {"extra_charts": [{"title": "Duplicate", "series": [
        {"key": "motion_vector_change_mean", "label": "Change", "color": "#123456"},
    ] * 2}]},
    {"extra_table_columns": [{"key": "absent", "label": "Missing", "digits": 2}]},
    {"extra_table_columns": [{"key": "participant_id", "label": "Person", "digits": 2}]},
    {"extra_table_columns": [{"key": "motion_vector_change_mean", "label": "", "digits": 2}]},
    {"extra_table_columns": [{"key": "motion_vector_change_mean", "label": "Change",
                               "digits": True}]},
    {"extra_table_columns": [{"key": "motion_vector_change_mean", "label": "Change",
                               "digits": 13}]},
    {"extra_table_columns": [{"key": "motion_vector_change_mean", "label": "Change",
                               "digits": 2}] * 2},
])
def test_visualizer_rejects_malformed_optional_catalogs(catalog: dict[str, Any]) -> None:
    """Reject unsafe IDs, nonnumeric fields, duplicate keys, and invalid formatting.

    Args:
        catalog (dict[str, Any]): Invalid optional visualization metadata.
    """
    with pytest.raises(ValueError):
        AuditVisualization(_rows(), catalog)


def test_optional_dashboard_views_render_offline_and_escape_source_text(tmp_path: Path) -> None:
    """Render optional data, gaps, missingness, and inert source labels without a browser.

    Args:
        tmp_path (Path): Temporary standalone dashboard directory.
    """
    import shutil
    import subprocess

    attack = "</script><img src=x onerror=alert(1)>"
    summary = _extra_summary()
    summary["extra_charts"][0]["title"] = attack
    summary["extra_charts"][0]["series"][0]["label"] = attack
    summary["extra_table_columns"][0]["label"] = attack
    document_path = AuditVisualization(_rows(), summary).write_html(tmp_path / "optional.html")
    document = document_path.read_text()
    assert attack not in document
    assert "innerHTML" not in document
    assert "initializeExtraViews();" in document
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional for offline JavaScript contract checks")
    harness = Path(__file__).parent / "fixtures/audit_dashboard_contract.js"
    completed = subprocess.run(
        [node, str(harness), str(document_path)], capture_output=True, text=True, timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert "contracts passed" in completed.stdout


def test_visualizer_rejects_text_and_boolean_optional_evidence() -> None:
    """Prevent display metadata from turning strings or flags into numeric evidence."""
    for values in (["1"] * 8, [True] * 8):
        rows = _rows()
        rows["nonnumeric"] = values
        with pytest.raises(ValueError):
            AuditVisualization(rows, {"extra_table_columns": [
                {"key": "nonnumeric", "label": "Nonnumeric", "digits": 2},
            ]})


def test_distribution_titles_wrap_long_feature_definitions() -> None:
    """Keep descriptive catalog titles inside their plot panels."""
    import matplotlib.pyplot as plt

    summary = _extra_summary()
    title = (
        "Mean axis-vector change between supported neighboring fixed-grid points in source units"
    )
    summary["feature_definitions"]["motion_vector_change_mean"] = title
    figure = AuditVisualization(_rows(), summary).plot_distributions()
    rendered = figure.axes[1].get_title()

    assert " ".join(rendered.splitlines()) == title
    assert max(map(len, rendered.splitlines())) <= 46
    plt.close(figure)
