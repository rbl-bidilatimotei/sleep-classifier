import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from statistics import mean, stdev

import yaml

from sleep_classifier.gru_training import GRUExperiment, TrainingConfig
from sleep_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
ALLOWED_OVERRIDES = {"context_epochs", "learning_rate", "patience", "hidden_size"}


class GRUExperimentSuite:

    def __init__(self, config_path: Path) -> None:
        """Initialize the GRU experiment suite.

        Args:
            config_path (Path): Path to the YAML configuration file for the experiment suite.
        """
        self.config_path = config_path.resolve()
        settings = yaml.safe_load(self.config_path.read_text())
        self.base_path = (self.config_path.parent / settings["base_config"]).resolve()
        self.output_dir = (self.config_path.parent / settings["output_dir"]).resolve()
        self.base = yaml.safe_load(self.base_path.read_text())

        TrainingConfig(self.base)

        recipes = self._validate_trials(settings["trials"])
        self.seeds = self._validate_seeds(settings.get("seeds"))
        self.trials = self._expand_seeds(recipes) if self.seeds else recipes


    def _validate_seeds(self, seeds: list[int] | None) -> list[int] | None:
        """Require a small set of distinct nonnegative training seeds.

        Args:
            seeds (list[int] | None): Distinct nonnegative training seeds.

        Raises:
            ValueError: If seeds are invalid or duplicated.
        """
        if seeds is None:
            return None
        if not isinstance(seeds, list) or not 2 <= len(seeds) <= 5:
            raise ValueError("Use two to five shared seeds")
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Seeds must be nonnegative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Seeds must be unique")

        return seeds


    def _expand_seeds(self, recipes: list[dict[str, object]]) -> list[dict[str, object]]:
        """Pair every recipe with the same initialization and loader seeds.

        Args:
            recipes (list[dict[str, object]]): Validated comparison recipes.

        Returns:
            list[dict[str, object]]: Expanded list of recipes paired with each seed.
        """
        return [
            recipe
            | {"name": f"{recipe['name']}-seed-{seed}", "recipe": recipe["name"], "seed": seed}
            for seed in self.seeds
            for recipe in recipes
        ]


    def _validate_trials(self, trials: list[dict[str, object]]) -> list[dict[str, object]]:
        """Require one reference and at most four single-setting comparisons.

        Args:
            trials (list[dict[str, object]]): Named recipes and their optional overrides.

        Returns:
            list[dict[str, object]]: Validated list of trials.

        Raises:
            ValueError: If trials violate the comparison policy.
        """
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


    def plan(self) -> dict[str, object]:
        """Describe trials without loading participant data or creating outputs.

        Returns:
            dict[str, object]: A dictionary describing the planned trials and related configuration.
        """
        return {
            "base_config": str(self.base_path),
            "output_dir": str(self.output_dir),
            "requested_device": self.base.get("device", "auto"),
            "seed": self.base["seed"],
            "shared_seeds": self.seeds,
            "maximum_epochs": sum(
                (self.base | trial.get("overrides", {}))["epochs"] for trial in self.trials
            ),
            "trials": self.trials,
        }


    def run(self) -> Path:
        """Run all trials sequentially and preserve partial results after failures.

        Returns:
            Path: The path to the summary file containing the results of all trials.
        """
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

        if self.seeds:
            summary["seed_comparison"] = self._summarize_seeds(summary["results"])

        return self._save_summary(summary)


    def _write_trial_config(self, trial: dict[str, object], device: str, config_dir: Path) -> Path:
        """Resolve source paths and isolate each trial's artifacts.

        Args:
            trial (dict[str, object]): The trial configuration.
            device (str): The device to run the trial on.
            config_dir (Path): The directory to store the trial's configuration.

        Returns:
            Path: The path to the written trial configuration file.
        """
        settings = self.base | trial.get("overrides", {})
        settings["prepare_config"] = str(
            (self.base_path.parent / self.base["prepare_config"]).resolve()
        )
        settings["output_dir"] = str(self.output_dir / trial["name"])
        settings["device"] = device
        settings["seed"] = trial.get("seed", self.base["seed"])
        config_path = config_dir / f"{trial['name']}.yaml"
        config_path.write_text(yaml.safe_dump(settings, sort_keys=False))
        return config_path


    def _read_result(self, trial: dict[str, object], metrics_path: Path) -> dict[str, object]:
        """Keep comparison scores and a hash of each complete metrics artifact.

        Args:
            trial (dict[str, object]): The trial configuration.
            metrics_path (Path): The path to the metrics file.

        Returns:
            dict[str, object]: A dictionary containing the trial's results and metadata.
        """
        content = metrics_path.read_bytes()
        metrics = json.loads(content)

        return {
            "name": trial["name"],
            "overrides": trial.get("overrides", {}),
            "recipe": trial.get("recipe", trial["name"]),
            "seed": trial.get("seed", self.base["seed"]),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "classes": metrics["classes"],
            "best_epoch": metrics["best_epoch"],
            "epochs_completed": len(metrics["history"]),
            "device": metrics["device"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": hashlib.sha256(content).hexdigest(),
        }

    def _summarize_seeds(self, results: list[dict[str, object]]) -> dict[str, object]:
        """Report recipe variation and paired macro-F1 differences across seeds.

        Args:
            results (list[dict[str, object]]): A list of trial results.

        Returns:
            dict[str, object]: Recipe aggregates and paired differences.
        """
        recipes = list(dict.fromkeys(result["recipe"] for result in results))
        scores = {
            recipe: {
                result["seed"]: result["macro_f1"]
                for result in results
                if result["recipe"] == recipe
            }
            for recipe in recipes
        }
        aggregates = {
            recipe: {
                "mean_macro_f1": mean(values.values()),
                "sample_std_macro_f1": stdev(values.values()),
                "min_macro_f1": min(values.values()),
                "max_macro_f1": max(values.values()),
            }
            for recipe, values in scores.items()
        }

        reference = recipes[0]
        paired = {}
        for recipe in recipes[1:]:
            differences = {
                str(seed): scores[recipe][seed] - scores[reference][seed] for seed in self.seeds
            }
            paired[recipe] = {
                "reference": reference,
                "macro_f1_differences": differences,
                "mean_difference": mean(differences.values()),
                "sample_std_difference": stdev(differences.values()),
            }

        return {"recipes": aggregates, "paired": paired}


    def _save_summary(self, summary: dict[str, object]) -> Path:
        """Replace the summary atomically after each completed trial.

        Args:
            summary (dict[str, object]): The summary to save.

        Returns:
            Path: The path to the saved summary file.
        """
        path = self.output_dir / "summary.json"

        temporary = self.output_dir / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(path)

        return path


def main() -> None:
    """Run the configured command-line operation."""
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
