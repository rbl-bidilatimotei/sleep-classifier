import argparse
import copy
import hashlib
import json
import logging
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn

from sleep_classifier.gru_model import CausalGRUClassifier
from sleep_classifier.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


class OpalSmokeExperiment:
    """Compare Adam with OPAL's offline keep backend on synthetic sequences."""

    def __init__(self, output_dir: Path, steps: int = 100, device: str = "cpu") -> None:
        if steps < 2:
            raise ValueError("steps must be at least two")
        if device not in ("cpu", "mps"):
            raise ValueError("device must be cpu or mps")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        self.output_dir = output_dir
        self.steps = steps
        self.device = torch.device(device)

    def run(self) -> dict[str, object]:
        """Check loss and parameter agreement without data files or agent calls.

        Returns:
            dict[str, object]: Loss comparison, parameter hashes, and timings.
        """
        # Keep OPAL optional for module imports and ordinary tests
        import opal

        self.output_dir.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(42)

        model = CausalGRUClassifier(hidden_size=8)
        initial_state = copy.deepcopy(model.state_dict())
        generator = torch.Generator().manual_seed(123)

        batches = self._make_batches(generator)
        plain = self._run_trial(initial_state, batches)
        wrapped = self._run_trial(initial_state, batches, opal)
        summary = self._compare(plain, wrapped)

        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        LOGGER.info(
            "Offline OPAL check passed: device=%s steps=%d max_loss_difference=%g output=%s",
            self.device,
            self.steps,
            summary["max_loss_difference"],
            self.output_dir,
        )
        return summary

    def _make_batches(self, generator: torch.Generator) -> list[tuple]:
        """Generate synthetic batches for the experiment.

        Args:
            generator (torch.Generator): Random number generator for reproducibility.

        Returns:
            list[tuple]: Synthetic features, lengths, and targets.
                - features (torch.Tensor): Features with batch, sequence, and feature dimensions.
                - lengths (torch.Tensor): Sequence lengths for each batch element.
                - targets (torch.Tensor): Target labels for each batch element.
        """
        batches = []
        for _ in range(self.steps):
            features = torch.randn(8, 5, 6, generator=generator).to(self.device)
            lengths = torch.tensor([1, 2, 3, 4, 5, 3, 2, 5])
            targets = torch.randint(0, 4, (8,), generator=generator).to(self.device)
            batches.append((features, lengths, targets))

        return batches

    def _run_trial(
        self, initial_state: dict[str, torch.Tensor], batches: list[tuple], opal=None
    ) -> dict[str, Any]:
        """Run a single training trial with the given model state and batches.

        Args:
            initial_state (dict[str, torch.Tensor]): Initial state of the model.
            batches (list[tuple]): Synthetic features, lengths, and targets.
            opal (ModuleType | None): Optional wrapper module.

        Returns:
            dict[str, Any]: Dictionary containing the results of the trial, including:
                - losses (list[float]): List of loss values for each batch.
                - state (dict[str, torch.Tensor]): Final state of the model.
                - elapsed_seconds (float): Time taken for the trial in seconds.
        """
        torch.manual_seed(42)
        model = CausalGRUClassifier(hidden_size=8).to(self.device)
        model.load_state_dict(initial_state)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if opal is not None:
            optimizer = opal.wrap(
                optimizer,
                model,
                total_steps=self.steps,
                goal="Synthetic four-class GRU; verify unchanged Adam updates",
                backend="keep",
                summarize=False,
                share_code=False,
                decisions=1,
                every=1,
                run_dir=str(self.output_dir / "opal"),
            )
        started = perf_counter()
        try:
            losses = self._updates(model, optimizer, batches, wrapped=opal is not None)
        finally:
            if opal is not None:
                optimizer.close()
        return {
            "losses": losses,
            "state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "elapsed_seconds": perf_counter() - started,
        }

    def _updates(self, model, optimizer, batches: list[tuple], wrapped: bool) -> list[float]:
        """Perform model updates for the given batches.

        Args:
            model (nn.Module): The model to be updated.
            optimizer (torch.optim.Optimizer): The optimizer used for updates.
            batches (list[tuple]): Synthetic features, lengths, and targets.
            wrapped (bool): Indicates if the optimizer is wrapped by OPAL.

        Returns:
            list[float]: List of loss values for each batch.
        """
        losses = []
        criterion = nn.CrossEntropyLoss()
        for features, lengths, targets in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features, lengths), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if wrapped:
                optimizer.step(loss=loss)
            else:
                optimizer.step()
            losses.append(float(loss.item()))
        return losses

    def _compare(self, plain: dict[str, Any], wrapped: dict[str, Any]) -> dict[str, Any]:
        """Compare the results of plain and wrapped optimizer runs.

        Args:
            plain (dict[str, Any]): Results from the plain optimizer run.
            wrapped (dict[str, Any]): Results from the wrapped optimizer run.

        Returns:
            dict[str, Any]: Comparison metrics and information.
        """
        absolute_tolerance = 1e-6 if self.device.type == "mps" else 0.0
        relative_tolerance = 1e-5 if self.device.type == "mps" else 0.0

        plain_losses = torch.tensor(plain["losses"], dtype=torch.float64)
        wrapped_losses = torch.tensor(wrapped["losses"], dtype=torch.float64)
        torch.testing.assert_close(
            plain_losses, wrapped_losses, atol=absolute_tolerance, rtol=relative_tolerance
        )
        torch.testing.assert_close(
            plain["state"], wrapped["state"], atol=absolute_tolerance, rtol=relative_tolerance
        )

        return {
            "backend": "keep",
            "agent_calls": 0,
            "synthetic_only": True,
            "steps": self.steps,
            "device": str(self.device),
            "torch_version": torch.__version__,
            "opal_version": version("opal-optimizer"),
            "max_loss_difference": float((plain_losses - wrapped_losses).abs().max()),
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "plain_parameter_sha256": self._parameter_digest(plain["state"]),
            "wrapped_parameter_sha256": self._parameter_digest(wrapped["state"]),
            "plain_seconds_per_step": plain["elapsed_seconds"] / self.steps,
            "wrapped_seconds_per_step": wrapped["elapsed_seconds"] / self.steps,
            "plain_losses": plain["losses"],
            "wrapped_losses": wrapped["losses"],
        }

    @staticmethod
    def _parameter_digest(state: dict) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(state.items()):
            digest.update(name.encode())
            digest.update(value.contiguous().numpy().tobytes())
        return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check OPAL with synthetic data and no agent")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    configure_logging()
    OpalSmokeExperiment(args.output_dir, args.steps, args.device).run()


if __name__ == "__main__":
    main()
