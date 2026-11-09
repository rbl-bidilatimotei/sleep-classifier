import argparse
import json
import logging
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from sleep_classifier.logging_utils import configure_logging
from sleep_classifier.paired_architecture_report import PairedArchitectureReport


class BudgetSeedReport:
    """Describe matched budget effects across shared training seeds."""

    def __init__(self, config_path: Path) -> None:
        """Resolve explicit grouped configurations without training or creating files.

        Args:
            config_path (Path): YAML study configuration with seed pairs.
        """
        self.config_path = config_path.resolve()
        settings = yaml.safe_load(self.config_path.read_text())
        pairs = settings["pairs"]
        if not isinstance(pairs, list) or not 2 <= len(pairs) <= 5:
            raise ValueError("Use two to five shared seed pairs")
        seeds = [pair["seed"] for pair in pairs]
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Training seeds must be nonnegative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Training seeds must be unique")

        self.output_dir = (self.config_path.parent / settings["output_dir"]).resolve()
        self.cases = [self._resolve_case(pair) for pair in pairs]
        outputs = [case[f"{role}_summary"] for case in self.cases
                   for role in ("reference", "candidate")]
        if len(set(outputs)) != len(outputs):
            raise ValueError("Every seed and budget requires a separate summary")
        if str(self.output_dir / "summary.json") in outputs:
            raise ValueError("Study output must not replace a grouped summary")


    def _resolve_case(self, pair: dict[str, Any]) -> dict[str, Any]:
        """Read the two explicit recipes and check their training seed.

        Args:
            pair (dict[str, Any]): Seed, reuse flag, and grouped configuration paths.
        """
        if type(pair["reuse"]) is not bool:
            raise ValueError("reuse must be a boolean")
        case = {"seed": pair["seed"], "reuse": pair["reuse"]}
        for role in ("reference", "candidate"):
            path = (self.config_path.parent / pair[f"{role}_config"]).resolve()
            grouped = yaml.safe_load(path.read_text())
            model_path = (path.parent / grouped["base_model_config"]).resolve()
            model = yaml.safe_load(model_path.read_text())
            if model["seed"] != pair["seed"]:
                raise ValueError("Grouped model seed differs from the study pair")
            case[f"{role}_config"] = str(path)
            case[f"{role}_summary"] = str(
                (path.parent / grouped["output_dir"] / "summary.json").resolve()
            )
            case[f"{role}_maximum_epochs"] = grouped["folds"] * model["epochs"]
            case[f"{role}_folds"] = grouped["folds"]

        return case


    def plan(self) -> dict[str, Any]:
        """Describe new work separately from reused results without optimization."""
        new = [case for case in self.cases if not case["reuse"]]

        return {
            "cases": self.cases,
            "new_fits": sum(case[f"{role}_folds"] for case in new
                            for role in ("reference", "candidate")),
            "maximum_new_epochs": sum(case[f"{role}_maximum_epochs"] for case in new
                                      for role in ("reference", "candidate")),
        }


    def status(self) -> list[dict[str, Any]]:
        """Distinguish unstarted pairs from complete or failed grouped runs."""
        rows = []
        for case in self.cases:
            for role in ("reference", "candidate"):
                path = Path(case[f"{role}_summary"])
                saved = json.loads(path.read_text()) if path.exists() else {}
                rows.append({"seed": case["seed"], "recipe": role, "reuse": case["reuse"],
                             "status": saved.get("status", "not_started"),
                             "completed_folds": len(saved.get("results", []))})

        return rows


    @staticmethod
    def _validate_seed_run(
        summary: dict[str, Any], summary_path: Path, seed: int,
        expected: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Require the same fold population and recipe apart from training seed.

        Args:
            summary (dict[str, Any]): Grouped summary already checked by the paired reporter.
            summary_path (Path): Location used to resolve relative metric paths.
            seed (int): Expected training seed for both budgets.
            expected (dict[str, Any] | None): First seed's canonical plan and recipe.
        """
        plan = summary["plan"]
        if plan["training_seed"] != seed:
            raise ValueError("Recorded training seed differs from the study pair")
        canonical_plan = {key: value for key, value in plan.items() if key != "training_seed"}
        recipe = None
        for result in summary["results"]:
            path = Path(result["metrics_path"])
            if not path.is_absolute():
                path = summary_path.parent / path
            metrics = json.loads(path.read_text())
            if metrics["config"]["seed"] != seed:
                raise ValueError("Fold training seed differs from the study pair")
            current = {
                "config": {key: value for key, value in metrics["config"].items()
                           if key not in ("seed", "prepare_config", "output_dir")},
                "model_config": metrics["model_config"],
                "parameter_count": metrics["parameter_count"],
                "device": metrics["device"],
            }
            if recipe is not None and recipe != current:
                raise ValueError("Training recipes differ across folds")
            recipe = current
        signature = {"plan": canonical_plan, "recipe": recipe}
        if expected is not None and signature != expected:
            raise ValueError("Folds or training recipes differ across seeds")

        return signature


    def build(self) -> dict[str, Any]:
        """Validate every complete pair and summarize seed-level effects descriptively."""
        reports = []
        signatures = {}
        for case in self.cases:
            paths = {role: Path(case[f"{role}_summary"])
                     for role in ("reference", "candidate")}
            report = PairedArchitectureReport(
                paths["reference"], paths["candidate"], comparison="training_budget",
            ).build()
            for role, path in paths.items():
                saved = json.loads(path.read_text())
                if (saved["plan"]["maximum_epochs"] != case[f"{role}_maximum_epochs"]
                        or len(saved["plan"]["folds"]) != case[f"{role}_folds"]):
                    raise ValueError("Recorded work differs from the configured seed study")
                signatures[role] = self._validate_seed_run(
                    saved, path, case["seed"], signatures.get(role),
                )
            reports.append({"seed": case["seed"], "report": report})

        effects = [entry["report"]["paired"]["mean_macro_f1_difference"] for entry in reports]
        return {
            "seeds": reports,
            "mean_seed_macro_f1_difference": mean(effects),
            "sample_std_seed_macro_f1_difference": stdev(effects),
            "independent_outer_test": False,
            "interpretation": (
                "Descriptive shared-seed development effects; the same participants repeat. "
                "Seed/fold cells are not independent populations or significance evidence."
            ),
        }


def main() -> None:
    """Plan a seed study or write its report without starting model training."""
    parser = argparse.ArgumentParser(description="Report paired training-budget seed effects")
    parser.add_argument("--config", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--plan", action="store_true")
    operation.add_argument("--report", action="store_true")
    args = parser.parse_args()
    configure_logging()
    study = BudgetSeedReport(args.config)
    if args.plan:
        logging.getLogger(__name__).info("Seed study plan: %s", json.dumps(study.plan()))
    else:
        report = study.build()
        study.output_dir.mkdir(parents=True, exist_ok=True)
        path = study.output_dir / "summary.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        logging.getLogger(__name__).info("Seed study report saved: %s", path)


if __name__ == "__main__":
    main()
