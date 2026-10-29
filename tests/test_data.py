"""Contracts for raw ingestion, provenance, and portable configuration."""

import hashlib
import io
from pathlib import Path

import pandas as pd
import pytest

from sleep_classifier.data import fetch_file, load_config, load_sensor, summarize


def test_config_paths_are_relative_to_yaml(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text('subject_id: "123"\ndata_dir: raw\nreport_path: report.json\n')
    monkeypatch.chdir(tmp_path.parent)
    assert load_config(config)["data_dir"] == tmp_path / "raw"


@pytest.mark.parametrize("subject", ["123", '"../123"'])
def test_invalid_subject_configuration(tmp_path, subject):
    config = tmp_path / "config.yaml"
    config.write_text(f"subject_id: {subject}\ndata_dir: raw\nreport_path: report.json\n")
    with pytest.raises(ValueError, match="subject_id"):
        load_config(config)


@pytest.mark.parametrize("separator", [" ", ","])
def test_ingestion_preserves_order_and_audit_detects_duplicates(tmp_path, separator):
    path = tmp_path / "hr.txt"
    path.write_text("30 60\n0 70\n0 70\n60 80\n".replace(" ", separator))
    frame = load_sensor(path, "heart_rate")
    assert frame.time.tolist() == [30, 0, 0, 60]
    report = summarize(frame, "heart_rate")
    assert report["backward_jumps"] == 1
    assert report["duplicate_timestamps"] == 1
    assert report["duplicate_rows"] == 1
    assert report["conflicting_duplicate_timestamps"] == 0
    assert report["median_sorted_unique_interval"] == 30


def test_conflicting_readings_are_distinguished_from_exact_duplicates():
    frame = pd.DataFrame({"time": [0, 0, 30], "heart_rate": [60, 70, 80]})
    report = summarize(frame, "heart_rate")
    assert report["conflicting_duplicate_timestamps"] == 1
    assert report["duplicate_rows"] == 0


@pytest.mark.parametrize("contents", ["0 60 2\n", "0 NaN\n", "0 inf\n"])
def test_malformed_observations_rejected(tmp_path, contents):
    path = tmp_path / "hr.txt"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_sensor(path, "heart_rate")


def test_unknown_labels_and_gaps_are_reported():
    frame = pd.DataFrame({"time": [0, 30, 90], "stage": [0, -1, 5]})
    report = summarize(frame, "labels")
    assert report["unexpected_labels"] == [-1]
    assert report["non_30_second_intervals"] == 1


def test_no_previous_day_steps_is_missing_coverage():
    frame = pd.DataFrame({"time": [-200000, -199400], "steps": [20, 30]})
    assert summarize(frame, "steps")["complete_bins_in_previous_24h"] == 0


def test_single_row_has_no_sampling_interval():
    frame = pd.DataFrame({"time": [0], "heart_rate": [60]})
    assert summarize(frame, "heart_rate")["median_sorted_unique_interval"] is None


def test_verified_download_and_offline_reuse(tmp_path, monkeypatch):
    contents = b"0 60\n"
    digest = hashlib.sha256(contents).hexdigest()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(contents))
    path = tmp_path / "hr.txt"
    fetch_file("https://example.test/data", path, digest)
    assert path.read_bytes() == contents

    def unexpected_network(*args, **kwargs):
        pytest.fail("Valid local files must not trigger network access")

    monkeypatch.setattr("urllib.request.urlopen", unexpected_network)
    fetch_file("https://example.test/data", path, digest)
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_file("https://example.test/data", path, digest)
    assert path.read_bytes() == b"corrupted"


def test_bad_download_does_not_publish_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    path = tmp_path / "hr.txt"
    with pytest.raises(ValueError, match="Downloaded checksum"):
        fetch_file("https://example.test/data", path, "0" * 64)
    assert not path.exists()
    assert not Path(str(path) + ".part").exists()
