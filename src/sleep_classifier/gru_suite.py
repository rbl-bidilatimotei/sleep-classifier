import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import yaml

from sleep_classifier.gru_training import GRUExperiment, TrainingConfig
from sleep_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
ALLOWED_OVERRIDES = {"context_epochs", "learning_rate", "patience", "hidden_size"}


class GRUExperimentSuite:
    """Run a bounded set of single-change GRU comparisons."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        settings = yaml.safe_load(self.config_path.read_text())
        self.base_path = (self.config_path.parent / settings["base_config"]).resolve()
        self.output_dir = (self.config_path.parent / settings["output_dir"]).resolve()
        self.base = yaml.safe_load(self.base_path.read_text())
        TrainingConfig(self.base)
        self.trials = self._validate_trials(settings["trials"])

    def _validate_trials(self, trials: list) -> list[dict]:
        """Require one reference and at most four single-setting comparisons."""
        if not isinstance(trials, list) or not 2 <= len(trials) <= 5:
            raise ValueError("A suite requires two to five trials")
        names = set()
        for trial in trials:
            name = trial["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
                raise ValueError("Trial names must be safe directory names")
            if name in names:
                raise ValueError("Trial names must be unique")
            names.add(name)
            overrides = trial.get("overrides", {})
            if not isinstance(overrides, dict) or len(overrides) > 1:
                raise ValueError("Each comparison can change only one setting")
            if set(overrides) - ALLOWED_OVERRIDES:
                raise ValueError("Trial overrides must preserve the device, data, and seed")
            if any(self.base[key] == value for key, value in overrides.items()):
                raise ValueError("A comparison must change its setting")
            TrainingConfig(self.base | overrides)
        if trials[0].get("overrides", {}) or any(
            not trial.get("overrides", {}) for trial in trials[1:]
        ):
            raise ValueError("Only the first trial must be the unchanged reference")
        return trials

    def plan(self) -> dict:
        """Describe trials without loading participant data or creating outputs."""
        return {
            "base_config": str(self.base_path),
            "output_dir": str(self.output_dir),
            "requested_device": self.base.get("device", "auto"),
            "seed": self.base["seed"],
            "maximum_epochs": sum(
                (self.base | trial.get("overrides", {}))["epochs"] for trial in self.trials
            ),
            "trials": self.trials,
        }

    def run(self) -> Path:
        """Run all trials sequentially and preserve partial results after failures."""
        device = str(TrainingConfig(self.base).device())
        self.output_dir.mkdir(parents=True, exist_ok=False)
        config_dir = self.output_dir / "configs"
        config_dir.mkdir()
        summary = self.plan() | {"device": device, "results": [], "status": "running"}
        self._save_summary(summary)
        for index, trial in enumerate(self.trials, start=1):
            LOGGER.info("Starting trial %d/%d: %s", index, len(self.trials), trial["name"])
            config_path = self._write_trial_config(trial, device, config_dir)
            try:
                metrics_path = GRUExperiment(config_path).run()
                summary["results"].append(self._read_result(trial, metrics_path))
            except Exception:
                summary.update(status="failed", failed_trial=trial["name"])
                self._save_summary(summary)
                raise
            self._save_summary(summary)
        summary["status"] = "complete"
        return self._save_summary(summary)

    def _write_trial_config(self, trial: dict, device: str, config_dir: Path) -> Path:
        """Resolve source paths and isolate each trial's artifacts."""
        settings = self.base | trial.get("overrides", {})
        settings["prepare_config"] = str(
            (self.base_path.parent / self.base["prepare_config"]).resolve()
        )
        settings["output_dir"] = str(self.output_dir / trial["name"])
        settings["device"] = device
        config_path = config_dir / f"{trial['name']}.yaml"
        config_path.write_text(yaml.safe_dump(settings, sort_keys=False))
        return config_path

    def _read_result(self, trial: dict, metrics_path: Path) -> dict:
        """Keep comparison scores and a hash of each complete metrics artifact."""
        content = metrics_path.read_bytes()
        metrics = json.loads(content)
        return {
            "name": trial["name"],
            "overrides": trial.get("overrides", {}),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "classes": metrics["classes"],
            "best_epoch": metrics["best_epoch"],
            "epochs_completed": len(metrics["history"]),
            "device": metrics["device"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": hashlib.sha256(content).hexdigest(),
        }

    def _save_summary(self, summary: dict) -> Path:
        """Replace the summary atomically after each completed trial."""
        path = self.output_dir / "summary.json"
        temporary = self.output_dir / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(path)
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run bounded GRU comparisons")
    parser.add_argument("--config", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--plan", action="store_true", help="Inspect settings without training")
    operation.add_argument("--train", action="store_true", help="Run the declared training trials")
    args = parser.parse_args()
    configure_logging()
    suite = GRUExperimentSuite(args.config)
    if args.plan:
        LOGGER.info("GRU suite plan: %s", json.dumps(suite.plan(), indent=2))
    else:
        LOGGER.info("GRU suite summary saved: %s", suite.run())


if __name__ == "__main__":
    main()
