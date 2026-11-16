import copy
import math
from typing import Any

import torch
from torch import nn

from sleep_classifier.features import FEATURE_COLUMNS
from sleep_classifier.model_factory import ModelFactory
from sleep_classifier.optimizer_factory import OptimizerFactory


class ZeroDecayControl:
    """Compare legacy Adam and grouped zero-decay AdamW on synthetic updates."""

    def __init__(
        self, settings: dict[str, Any], device: torch.device | str,
        steps: int = 20, seed: int = 42,
    ) -> None:
        """Configure a bounded synthetic check without reading participant data.

        Args:
            settings (dict[str, Any]): Candidate architecture and training settings.
            device (torch.device | str): Device used by both optimizer paths.
            steps (int): Number of synthetic updates, from one through one hundred.
            seed (int): Nonnegative seed for model, batch, and dropout draws.
        """
        if type(steps) is not int or not 1 <= steps <= 100:
            raise ValueError("Synthetic control steps must be an integer in [1, 100]")
        if type(seed) is not int or seed < 0:
            raise ValueError("Synthetic control seed must be a nonnegative integer")

        rate = settings.get("learning_rate")
        clip = settings.get("gradient_clip")
        for name, value in (("learning_rate", rate), ("gradient_clip", clip)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

        self.settings = copy.deepcopy(settings)
        self.device = torch.device(device)
        self.steps = steps
        self.seed = seed
        self.relative_tolerance = 1e-6
        self.absolute_tolerance = 1e-7


    def run(self) -> dict[str, Any]:
        """Check clipped updates and optimizer moments while restoring random states."""
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                return self._run_synthetic_updates()
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


    def _run_synthetic_updates(self) -> dict[str, Any]:
        """Construct identical models and compare the configured synthetic steps."""
        torch.manual_seed(self.seed)
        reference = ModelFactory(self.settings).build(len(FEATURE_COLUMNS), 4)
        reference = reference.to(self.device).train()
        candidate = copy.deepcopy(reference)
        adam = OptimizerFactory(None).create(reference, float(self.settings["learning_rate"]))
        factory = OptimizerFactory({
            "name": "adamw", "weight_decay": 0.0, "exclude_bias_and_norm": True,
        })
        adamw = factory.create(candidate, float(self.settings["learning_rate"]))
        generator = torch.Generator().manual_seed(self.seed + 1)
        maxima = dict.fromkeys(("loss", "gradient", "parameter", "optimizer_state"), 0.0)
        passed = True

        for step in range(self.steps):
            features, lengths, targets = self._synthetic_batch(generator)
            losses = []
            for model, optimizer in ((reference, adam), (candidate, adamw)):
                torch.manual_seed(self.seed + 100 + step)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(features, lengths), targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(self.settings["gradient_clip"]))
                optimizer.step()
                losses.append(loss.detach())

            differences = self._step_differences(reference, candidate, adam, adamw, losses)
            for name, (difference, close) in differences.items():
                maxima[name] = max(maxima[name], difference)
                passed = passed and close

        return {
            "passed": passed, "exact_match": all(value == 0.0 for value in maxima.values()),
            "device": str(self.device), "steps": self.steps, "seed": self.seed,
            "torch_version": torch.__version__, "synthetic_only": True,
            "participant_data_read": False,
            "dropout_enabled": any(
                isinstance(module, nn.Dropout) and module.p > 0 for module in reference.modules()
            ),
            "rtol": self.relative_tolerance, "atol": self.absolute_tolerance,
            **{f"max_{name}_difference": value for name, value in maxima.items()},
            "groups": factory.describe_groups(candidate),
            "interpretation": (
                "Bounded synthetic numerical control; not a proof of equal participant trajectories"
            ),
        }


    def _synthetic_batch(
        self, generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create variable-length histories with no participant observations.

        Args:
            generator (torch.Generator): Isolated CPU generator for synthetic batches.
        """
        context = int(self.settings["context_epochs"])
        lengths = torch.tensor([
            context, max(1, context - 1), max(1, context // 2), max(1, context // 3),
            min(5, context), min(3, context), min(2, context), 1,
        ])
        features = torch.randn(8, context, len(FEATURE_COLUMNS), generator=generator)
        features = features.to(self.device)
        targets = torch.randint(4, (8,), generator=generator).to(self.device)
        return features, lengths, targets


    def _step_differences(
        self, reference: nn.Module, candidate: nn.Module,
        adam: torch.optim.Optimizer, adamw: torch.optim.Optimizer,
        losses: list[torch.Tensor],
    ) -> dict[str, tuple[float, bool]]:
        """Compare clipped gradients, updated weights, losses, and Adam moments.

        Args:
            reference (nn.Module): Legacy Adam model after its update.
            candidate (nn.Module): Grouped zero-decay AdamW model after its update.
            adam (torch.optim.Optimizer): Reference state indexed by reference parameters.
            adamw (torch.optim.Optimizer): Candidate state indexed by candidate parameters.
            losses (list[torch.Tensor]): The two pre-update cross-entropy losses.
        """
        pairs = list(zip(reference.parameters(), candidate.parameters(), strict=True))
        reference_state = []
        candidate_state = []
        for left, right in pairs:
            if adam.state[left].keys() != adamw.state[right].keys():
                raise ValueError("Zero-decay optimizer state schemas differ")
            for key in adam.state[left]:
                reference_state.append(adam.state[left][key])
                candidate_state.append(adamw.state[right][key])

        return {
            "loss": self._compare_tensors(losses[:1], losses[1:]),
            "parameter": self._compare_tensors([p for p, _ in pairs], [p for _, p in pairs]),
            "gradient": self._compare_tensors(
                [p.grad for p, _ in pairs if p.grad is not None],
                [p.grad for _, p in pairs if p.grad is not None],
            ),
            "optimizer_state": self._compare_tensors(reference_state, candidate_state),
        }


    def _compare_tensors(
        self, reference: list[torch.Tensor], candidate: list[torch.Tensor],
    ) -> tuple[float, bool]:
        """Return the largest absolute difference and elementwise tolerance result.

        Args:
            reference (list[torch.Tensor]): Reference values in parameter order.
            candidate (list[torch.Tensor]): Candidate values in the same order.
        """
        maximum = 0.0
        passed = True
        for left, right in zip(reference, candidate, strict=True):
            finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            difference = float((left.detach() - right.detach()).abs().max()) if finite else math.inf
            maximum = max(maximum, difference)
            passed = passed and finite and torch.allclose(
                left, right, rtol=self.relative_tolerance, atol=self.absolute_tolerance,
            )
        return maximum, bool(passed)


def main() -> None:
    """Run the bounded synthetic control from a saved training configuration."""
    import argparse
    import json
    import logging
    from pathlib import Path

    import yaml

    from sleep_classifier.logging_utils import configure_logging
    from sleep_classifier.neural_training import TrainingConfig

    parser = argparse.ArgumentParser(
        description="Verify Adam and zero-decay AdamW with synthetic data",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.resolve() == args.config.resolve():
        parser.error("The output must not overwrite the training configuration")
    configure_logging()
    settings = yaml.safe_load(args.config.read_text())
    if args.device is not None:
        settings["device"] = args.device
    configuration = TrainingConfig(settings)
    result = ZeroDecayControl(
        settings, configuration.device(), steps=args.steps, seed=int(settings["seed"]),
    ).run()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    logging.getLogger(__name__).info("Synthetic zero-decay control: %s", json.dumps(result))
    if not result["passed"]:
        raise SystemExit("Zero-decay control failed; inspect the differences before training")


if __name__ == "__main__":
    main()
