import copy
import posixpath
import re
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import torch

from sleep_classifier.bundle_validation import CHECKPOINT_FIELDS, sha256_bytes, strict_json_loads
from sleep_classifier.causal_feature_audit import AuditSettings
from sleep_classifier.chronological_replay import ChronologicalReplayStudy


class EvaluationExportSource:
    """Verify the selected development checkpoint against completed chronological evidence."""

    def __init__(self, audit_config: Path, fold: str) -> None:
        """Bind the completed audit and an explicitly selected evaluation fold.

        Args:
            audit_config (Path): Frozen chronological audit configuration.
            fold (str): Numbered development fold to export without ranking folds.
        """
        self.study = ChronologicalReplayStudy(audit_config)
        self.fold = fold
        self.fold_dir = self.study.reference.output_dir / fold


    def protected_roots(self) -> tuple[Path, ...]:
        """Identify measured artifacts and raw/cache directories without reading their contents."""
        reference = self.study.reference
        roots = [self.study.output_dir, reference.output_dir, reference.source_dir]
        motion = reference.model_settings.get("motion_features")
        if motion:
            audit_path = (reference.model_path.parent / motion["audit_config"]).resolve()
            audit = AuditSettings(audit_path)
            roots.extend((audit.output_dir, audit.raw_data_dir))
        return tuple(path.resolve() for path in roots)


    def snapshot(self) -> dict[str, Any]:
        """Verify source identity and require the selected fold's full parity audit to pass."""
        report = self.study.read_report()
        if report["status"] != "passed" or self.fold not in report["plan"]["folds"]:
            raise ValueError("Export requires a passed chronological audit and a planned fold")
        if (report.get("training_performed") is not False
                or report.get("test_observations_read") is not False):
            raise ValueError("Export evidence must exclude training and final-test observations")
        row = next(item for item in report["folds"] if item["fold"] == self.fold)
        self._validate_parity(row)
        return {"plan": report["plan"], "fold": copy.deepcopy(row)}


    @staticmethod
    def _validate_parity(row: dict[str, Any]) -> None:
        """Require the original numerical, history, and model-state acceptance conditions.

        Args:
            row (dict[str, Any]): Registered selected-fold chronological evidence.
        """
        if (row["status"] != "passed" or row["device"] != "mps"
                or row["absolute_tolerance"] != 2e-6 or row["context_epochs"] != 30
                or row["history_lengths_match"] is not True or row["weights_unchanged"] is not True
                or row["gradients_created"] is not False or row["training_performed"] is not False):
            raise ValueError("Source fold does not meet the frozen parity policy")
        for name in ("stream_vs_offline", "offline_vs_saved"):
            values = row[name]
            difference = values["max_abs_probability_difference"]
            if (not isinstance(difference, (int, float)) or not 0 <= difference <= 2e-6
                    or values["prediction_disagreements"] != 0):
                raise ValueError("Source probabilities do not meet the frozen parity tolerance")


    def checkpoint(self, snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read verified weights and retain only fields required for inference.

        Args:
            snapshot (dict[str, Any]): Verified source identity before checkpoint reading.
        """
        directory = self.fold_dir / "results"
        expected = snapshot["plan"]["sources"][self.fold]
        data = (directory / "model_checkpoint.pt").read_bytes()
        digest = sha256_bytes(data)
        if (digest != expected["model_checkpoint.pt"]
                or digest != snapshot["fold"]["checkpoint_sha256"]):
            raise ValueError("Selected checkpoint bytes differ from the completed audit")
        metrics = self.metrics(snapshot)
        checkpoint = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
        self._match_metrics(checkpoint, metrics)
        if not CHECKPOINT_FIELDS <= checkpoint.keys():
            raise ValueError("Source checkpoint lacks required inference metadata")
        sanitized = {name: copy.deepcopy(checkpoint[name]) for name in sorted(CHECKPOINT_FIELDS)}
        return sanitized, metrics


    def metrics(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Read the registered source metadata without loading checkpoint tensors.

        Args:
            snapshot (dict[str, Any]): Verified source identity for the selected fold.
        """
        data = (self.fold_dir / "results/metrics.json").read_bytes()
        if sha256_bytes(data) != snapshot["plan"]["sources"][self.fold]["metrics.json"]:
            raise ValueError("Selected metrics bytes differ from the completed audit")
        metrics = strict_json_loads(data)
        prepared = strict_json_loads((self.fold_dir / "features/provenance.json").read_bytes())
        if (metrics.get("provenance") != prepared
                or prepared.get("test_observations_read") is not False):
            raise ValueError("Fold metrics differ from the prepared development provenance")
        return metrics


    @staticmethod
    def _match_metrics(checkpoint: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Link saved constructor, transforms, and training identity to registered metrics.

        Args:
            checkpoint (dict[str, Any]): Original selected training checkpoint.
            metrics (dict[str, Any]): Registered fold metrics from the chronological source.
        """
        if checkpoint.get("training_config") != metrics["config"]:
            raise ValueError("Selected checkpoint training configuration differs from metrics")
        for name in ("model_type", "model_config", "features", "model_features",
                     "motion_features", "window_feature_state"):
            if checkpoint.get(name) != metrics.get(name):
                raise ValueError(f"Selected checkpoint differs from metrics at {name}")
        if metrics["device"] != "mps" or metrics["config"]["learning_rate"] != 0.001:
            raise ValueError("Export requires the working LR0.001 MPS reference")


class GitBundleDocumentation:
    """Read committed documentation and make its local evidence links portable."""

    def __init__(self, repository: Path, revision: str) -> None:
        """Bind the approved source repository and exact runtime commit.

        Args:
            repository (Path): Source checkout containing the model card and lock file.
            revision (str): Exact runtime source revision from the identity guard.
        """
        self.repository = repository.resolve()
        self.revision = revision
        top = Path(self._git("rev-parse", "--show-toplevel").decode().strip()).resolve()
        if top != self.repository or self._git("rev-parse", "HEAD").decode().strip() != revision:
            raise ValueError("Documentation repository must match the current runtime revision")
        remote = self._git("remote", "get-url", "origin").decode().strip()
        if remote.startswith("git@github.com:"):
            remote = "https://github.com/" + remote.removeprefix("git@github.com:")
        parsed = urlsplit(remote)
        if parsed.hostname != "github.com" or parsed.scheme != "https":
            raise ValueError("Portable model-card links require the approved GitHub repository")
        self.repository_url = "https://github.com" + parsed.path.removesuffix(".git").rstrip("/")


    def _git(self, *arguments: str) -> bytes:
        """Read local Git objects without shell interpolation or network access.

        Args:
            *arguments (str): Git arguments relative to the declared source repository.
        """
        process = subprocess.run(["git", "-C", str(self.repository), *arguments],
                                 capture_output=True, check=False)
        if process.returncode:
            raise ValueError("Cannot read the declared documentation revision")
        return process.stdout


    def payload(self) -> dict[str, bytes]:
        """Return committed lock bytes and a model card with revision-pinned evidence URLs."""
        if self._git("status", "--porcelain=v1", "--", "docs/model-card.md").strip():
            raise ValueError("Commit model-card edits before exporting the documentation")
        card = self._git("show", f"{self.revision}:docs/model-card.md").decode("utf-8")
        card = re.sub(r"(!?\[[^\]]*\]\()([^)]+)(\))", self._rewrite_link, card)
        return {"MODEL_CARD.md": card.encode("utf-8"),
                "uv.lock": self._git("show", f"{self.revision}:uv.lock")}


    def _rewrite_link(self, match: re.Match[str]) -> str:
        """Rewrite one local model-card target after verifying its committed destination.

        Args:
            match (re.Match[str]): Markdown link or image target from the committed model card.
        """
        target = match.group(2)
        if target.startswith(("https://", "http://", "mailto:", "#")):
            return match.group(0)
        path, separator, fragment = target.partition("#")
        path = posixpath.normpath(posixpath.join("docs", path))
        if path == ".." or path.startswith(("../", "/")):
            raise ValueError("Model-card link escapes the source repository")
        self._git("cat-file", "-e", f"{self.revision}:{path}")
        url = f"{self.repository_url}/blob/{self.revision}/{quote(path, safe='/')}"
        if separator:
            url += "#" + fragment
        return match.group(1) + url + match.group(3)
