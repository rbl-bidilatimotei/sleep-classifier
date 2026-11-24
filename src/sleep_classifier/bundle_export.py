import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import yaml

from sleep_classifier.bundle_export_source import EvaluationExportSource, GitBundleDocumentation
from sleep_classifier.bundle_loader import EvaluationBundle
from sleep_classifier.bundle_parity import EvaluationBundleParity
from sleep_classifier.bundle_publication import publish_directory
from sleep_classifier.bundle_runtime import RUNTIME_PATHS, RuntimeIdentity
from sleep_classifier.bundle_validation import (
    CHECKPOINT_FIELDS,
    sha256_bytes,
    validate_checkpoint,
    validate_schema,
)
from sleep_classifier.logging_utils import configure_logging
from sleep_classifier.streaming_inference import StreamingEpochPredictor

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BundleExportConfig:
    """Resolve the explicit source fold and immutable export destinations."""

    audit_config: Path
    source_repository: Path
    output_dir: Path
    verification_dir: Path
    bundle_id: str
    fold: str

    @classmethod
    def load(cls, path: Path) -> "BundleExportConfig":
        """Read supported export settings relative to their YAML file.

        Args:
            path (Path): Export configuration file.
        """
        path = path.resolve()
        settings = yaml.safe_load(path.read_text())
        required = {"audit_config", "source_repository", "output_dir", "verification_dir",
                    "bundle_id", "fold"}
        if (not isinstance(settings, dict) or set(settings) != required
                or any(type(value) is not str or not value for value in settings.values())):
            raise ValueError("Export config must contain exactly the supported string settings")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", settings["bundle_id"]) is None:
            raise ValueError("Export requires a version-one bundle identifier")
        if re.fullmatch(r"fold-0[1-5]", settings["fold"]) is None:
            raise ValueError("Export requires an explicit planned evaluation fold")
        paths = {name: (path.parent / settings[name]).resolve() for name in (
            "audit_config", "source_repository", "output_dir", "verification_dir",
        )}
        return cls(**paths, bundle_id=settings["bundle_id"], fold=settings["fold"])


class EvaluationBundleExporter:
    """Publish one sanitized evaluation bundle only after full validation-fold parity."""

    def __init__(self, config_path: Path) -> None:
        """Bind a frozen source checkpoint and fresh output identifier.

        Args:
            config_path (Path): Export YAML with an explicitly selected development fold.
        """
        self.config_path = config_path.resolve()
        self.settings = BundleExportConfig.load(self.config_path)
        self.source = EvaluationExportSource(self.settings.audit_config, self.settings.fold)
        self.output_dir = self.settings.output_dir / self.settings.bundle_id
        self.verification_path = self.settings.verification_dir / f"{self.settings.bundle_id}.json"
        protected = (*self.source.protected_roots(),
                     (self.settings.source_repository / "data").resolve(),
                     *((self.settings.source_repository / path).resolve()
                       for path in RUNTIME_PATHS))
        if any(self.output_dir.is_relative_to(path) or self.verification_path.is_relative_to(path)
               for path in protected):
            raise ValueError(
                "Bundle destinations must stay outside source artifacts, data, and runtime",
            )
        if self.verification_path.is_relative_to(self.output_dir):
            raise ValueError("Verification evidence must stay outside the four-file bundle")


    def plan(self) -> dict[str, Any]:
        """Verify the source and runtime identity without restoring or running a model."""
        identity = RuntimeIdentity.current()
        GitBundleDocumentation(self.settings.source_repository, identity["source_revision"])
        return {"runtime": identity, "source": self.source.snapshot(),
                "bundle_id": self.settings.bundle_id, "fold": self.settings.fold,
                "output_dir": str(self.output_dir),
                "verification_path": str(self.verification_path)}


    def validate_start(self) -> dict[str, Any]:
        """Reject reused destinations and unavailable MPS before creating any output."""
        if self.output_dir.exists() or self.output_dir.is_symlink():
            raise FileExistsError(
                "Bundle identifier exists; preserve it and select a new identifier",
            )
        if self.verification_path.exists() or self.verification_path.is_symlink():
            raise FileExistsError("Bundle verification evidence already exists; preserve it")
        plan = self.plan()
        StreamingEpochPredictor._resolve_device("mps")
        return plan


    def read_existing(self) -> EvaluationBundle:
        """Verify a cached bundle against the active configuration and source."""
        bundle = EvaluationBundle(self.output_dir)
        plan = self.plan()
        manifest = bundle.manifest
        metrics = self.source.metrics(plan["source"])
        if (manifest["provenance"]["training_seed"] != metrics["config"]["seed"]
                or manifest["bundle_id"] != self.settings.bundle_id
                or manifest["runtime"] != (plan["runtime"] | {"validated_device": "mps"})
                or manifest["provenance"]["fold_id"] != self.settings.fold
                or manifest["provenance"]["source_checkpoint_sha256"]
                != plan["source"]["plan"]["sources"][self.settings.fold]["model_checkpoint.pt"]
                or manifest["provenance"]["metrics_sha256"]
                != plan["source"]["plan"]["sources"][self.settings.fold]["metrics.json"]
                or manifest["provenance"]["grouped_manifest_sha256"]
                != plan["source"]["plan"]["reference_manifest_sha256"]):
            raise ValueError("Cached bundle differs from the active export configuration or source")
        return bundle


    def export(self) -> Path:
        """Sanitize, verify, and publish a fresh bundle without changing the source checkpoint."""
        plan = self.validate_start()
        checkpoint, metrics = self.source.checkpoint(plan["source"])
        payload = GitBundleDocumentation(
            self.settings.source_repository, plan["runtime"]["source_revision"],
        ).payload()
        manifest = self._manifest(plan, checkpoint, metrics)
        validate_schema(manifest, "manifest")
        validate_checkpoint(checkpoint, manifest)
        payload["checkpoint.pt"] = self._checkpoint_bytes(checkpoint)
        for name, record in manifest["files"].items():
            record["sha256"] = sha256_bytes(payload[record["path"]])
        validate_schema(manifest, "manifest")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".bundle-stage-", dir=self.settings.output_dir) as temporary:
            staging = Path(temporary)
            self._write_staging(staging, payload, manifest)
            bundle = EvaluationBundle(staging)
            parity = EvaluationBundleParity(self.source.fold_dir, bundle).build()
            if parity.get("status") != "passed":
                raise ValueError("Bundle publication requires successful full-fold parity")
            if self.plan() != plan:
                raise ValueError("Source or runtime changed during export; no bundle was published")
            evidence = self._verification_bytes(plan, manifest, parity, staging)
            self._publish_with_evidence(staging, evidence)
        LOGGER.info("Published evaluation bundle: %s", self.output_dir)
        return self.output_dir


    def _manifest(
        self, plan: dict[str, Any], checkpoint: dict[str, Any], metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Describe the retained model and distinguish original from packaged checkpoint identity.

        Args:
            plan (dict[str, Any]): Verified source and runtime snapshot.
            checkpoint (dict[str, Any]): Sanitized inference metadata and CPU weights.
            metrics (dict[str, Any]): Source fold metrics matched to the training checkpoint.
        """
        source = plan["source"]["plan"]
        paths = {"checkpoint": "checkpoint.pt", "model_card": "MODEL_CARD.md",
                 "environment_lock": "uv.lock"}
        return {
            "schema_version": "1.0.0", "feature_contract_version": "1.0.0",
            "bundle_id": self.settings.bundle_id, "artifact_role": "evaluation_fold",
            "files": {name: {"path": path, "sha256": "0" * 64} for name, path in paths.items()},
            "model": {"model_type": checkpoint["model_type"], "context_epochs": 30,
                      "parameter_count": metrics["parameter_count"],
                      "raw_features": checkpoint["features"],
                      "effective_features": checkpoint["model_features"],
                      "class_names": ["wake", "light", "deep", "REM"]},
            "runtime": plan["runtime"] | {"validated_device": source["device"]},
            "provenance": {"fold_id": self.settings.fold,
                           "training_seed": metrics["config"]["seed"],
                           "grouped_manifest_sha256": source["reference_manifest_sha256"],
                           "metrics_sha256": source["sources"][self.settings.fold]["metrics.json"],
                           "source_checkpoint_sha256": source["sources"][self.settings.fold][
                               "model_checkpoint.pt"
                           ], "final_test_used": False},
        }


    @staticmethod
    def _checkpoint_bytes(checkpoint: dict[str, Any]) -> bytes:
        """Round-trip the sanitized checkpoint and require identical values and tensor bits.

        Args:
            checkpoint (dict[str, Any]): Validated inference-only checkpoint.
        """
        output = BytesIO()
        torch.save(checkpoint, output)
        data = output.getvalue()
        restored = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
        if not EvaluationBundleExporter._identical(checkpoint, restored):
            raise ValueError("Checkpoint serialization changed retained values or tensors")
        return data


    @staticmethod
    def _identical(left: Any, right: Any) -> bool:
        """Compare nested fitted values and exact tensor representations.

        Args:
            left (Any): Value before serialization.
            right (Any): Restored value after serialization.
        """
        if type(left) is not type(right):
            return False
        if isinstance(left, torch.Tensor):
            return (left.dtype == right.dtype and left.shape == right.shape
                    and torch.equal(left.reshape(-1).view(torch.uint8),
                                    right.reshape(-1).view(torch.uint8)))
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(
                EvaluationBundleExporter._identical(value, right[key])
                for key, value in left.items()
            )
        if isinstance(left, list):
            return len(left) == len(right) and all(
                EvaluationBundleExporter._identical(a, b) for a, b in zip(left, right, strict=True)
            )
        return bool(left == right)


    @staticmethod
    def _write_staging(
        staging: Path, payload: dict[str, bytes], manifest: dict[str, Any],
    ) -> None:
        """Assemble only the four contract files in a private staging directory.

        Args:
            staging (Path): Fresh private directory for acceptance checks.
            payload (dict[str, bytes]): Three sanitized or committed file payloads.
            manifest (dict[str, Any]): Validated manifest with packaged-file hashes.
        """
        for name, data in payload.items():
            (staging / name).write_bytes(data)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        )


    def _publish(self, staging: Path) -> None:
        """Atomically publish the complete bundle without replacing any existing identifier.

        Args:
            staging (Path): Fully verified private candidate directory.
        """
        publish_directory(staging, self.output_dir)


    def _publish_with_evidence(self, staging: Path, evidence: bytes) -> None:
        """Prepare evidence before atomic publication and roll back only this attempt's report.

        Args:
            staging (Path): Complete candidate on the destination filesystem.
            evidence (bytes): Serialized verification report prepared before publication.
        """
        self.settings.verification_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".bundle-evidence-", dir=self.settings.verification_dir,
        ) as tmp:
            report = Path(tmp) / "verification.json"
            report.write_bytes(evidence)
            os.link(report, self.verification_path)
            try:
                self._publish(staging)
            except BaseException:
                try:
                    owned = report.stat()
                    current = self.verification_path.lstat()
                    if (owned.st_dev, owned.st_ino) == (current.st_dev, current.st_ino):
                        self.verification_path.unlink()
                except FileNotFoundError:
                    pass
                raise


    def _verification_bytes(
        self, plan: dict[str, Any], manifest: dict[str, Any], parity: dict[str, Any], staging: Path,
    ) -> bytes:
        """Prepare external acceptance evidence before publishing the immutable bundle.

        Args:
            plan (dict[str, Any]): Pre-export source and runtime snapshot.
            manifest (dict[str, Any]): Published bundle manifest.
            parity (dict[str, Any]): Complete validation-fold comparison against saved predictions.
            staging (Path): Complete private candidate with the final manifest bytes.
        """
        evidence = {"status": "passed", "bundle_id": self.settings.bundle_id,
                    "manifest_sha256": sha256_bytes(
                        (staging / "manifest.json").read_bytes(),
                    ),
                    "runtime": manifest["runtime"], "source": plan["source"], "parity": parity,
                    "sanitized_checkpoint_fields": sorted(CHECKPOINT_FIELDS),
                    "training_performed": False, "test_observations_read": False}
        return (json.dumps(evidence, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main() -> None:
    """Export an explicitly selected evaluation fold with mandatory parity checks."""
    parser = argparse.ArgumentParser(
        description="Export a verified evaluation-fold inference bundle",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--export", action="store_true", required=True)
    args = parser.parse_args()
    configure_logging()
    LOGGER.info("Evaluation bundle saved: %s", EvaluationBundleExporter(args.config).export())


if __name__ == "__main__":
    main()
