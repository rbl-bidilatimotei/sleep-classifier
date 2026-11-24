from collections.abc import Callable
from pathlib import Path

import pytest
from test_grouped_validation import _configuration

from sleep_classifier.grouped_validation import GroupedValidationExperiment


class DigestReadGuard:
    """Observe the content-read boundary without opening an unexpected table."""

    def __init__(self, forbidden: Path, original: Callable[[Path], str]) -> None:
        """Retain the forbidden path and original content-hashing operation.

        Args:
            forbidden (Path): Unexpected input that must remain unopened.
            original (Callable[[Path], str]): Original content hashing function.
        """
        self.forbidden = forbidden
        self.original = original
        self.accesses: list[Path] = []


    def __call__(self, path: Path) -> str:
        """Reject content hashing of the unplanned input.

        Args:
            path (Path): Input requested by the manifest validator.
        """
        if path == self.forbidden:
            self.accesses.append(path)
            raise AssertionError("Unexpected observations were opened")
        return self.original(path)


@pytest.mark.parametrize("relative", ["features/test/reserved.csv", "unexpected.yaml"])
def test_unexpected_fold_inputs_are_rejected_before_content_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str,
) -> None:
    """Reject inventory drift without opening unexpected participant observations.

    Args:
        tmp_path (Path): Synthetic preparation directory.
        monkeypatch (pytest.MonkeyPatch): Scoped content-read guard.
        relative (str): Unexpected table or recipe path inside a prepared fold.
    """
    experiment = GroupedValidationExperiment(_configuration(tmp_path))
    experiment.prepare()
    fold = experiment.plan()["folds"][0]["name"]
    forbidden = experiment.output_dir / fold / relative
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("DO NOT READ THESE OBSERVATIONS")
    guard = DigestReadGuard(forbidden, experiment._digest)
    monkeypatch.setattr(experiment, "_digest", guard)
    with pytest.raises(ValueError, match="inventory"):
        experiment._validate_manifest()
    assert guard.accesses == []
