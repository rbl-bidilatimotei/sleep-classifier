from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from sleep_classifier.codex_runtime import CodexRuntime


class AgentOptimizerSettings:
    """Validate the bounded optimizer-agent pilot settings."""

    def __init__(self, settings: dict[str, Any] | None) -> None:
        """Initialize and validate the optimizer-agent settings.

        Args:
            settings (dict[str, Any] | None): Optimizer-agent settings to validate.
        """
        if settings is not None and not isinstance(settings, dict):
            raise ValueError("optimizer_agent must be a mapping or null")

        self.values = dict(settings or {})
        if not self.values:
            return

        allowed = {
            "backend",
            "goal",
            "decisions",
            "every",
            "model",
            "effort",
            "max_tokens",
            "binary",
            "schedule",
        }
        if set(self.values) - allowed:
            raise ValueError("Unknown optimizer_agent settings")
        if self.values.get("backend") not in ("keep", "codex"):
            raise ValueError("optimizer_agent backend must be keep or codex")
        if not isinstance(self.values.get("goal"), str) or not self.values["goal"].strip():
            raise ValueError("optimizer_agent goal must be nonempty")

        for name in ("decisions", "every"):
            value = self.values.get(name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"optimizer_agent {name} must be a positive integer")
        schedule = self.values.get("schedule")
        if schedule is not None:
            if (
                not isinstance(schedule, list)
                or len(schedule) != self.values["decisions"]
                or any(type(step) is not int or step <= 0 for step in schedule)
                or schedule != sorted(set(schedule))
                or any(step % self.values["every"] for step in schedule)
            ):
                raise ValueError(
                    "schedule must match decisions with increasing positive check updates"
                )
        if self.values["decisions"] > 3:
            raise ValueError("The pilot allows at most three decisions")

        if self.values["backend"] == "codex":
            self._validate_codex()
        elif set(self.values) & {"model", "effort", "max_tokens", "binary"}:
            raise ValueError("The keep backend does not accept model settings")


    def _validate_codex(self) -> None:
        """Require an explicit model and a soft token budget."""
        if not isinstance(self.values.get("model"), str) or not self.values["model"].strip():
            raise ValueError("Codex requires an explicit model")
        if self.values.get("effort") not in ("high", "xhigh"):
            raise ValueError("Codex effort must be high or xhigh")

        binary = self.values.get("binary")
        if binary is not None and (not isinstance(binary, str) or not binary.strip()):
            raise ValueError("Codex binary must be a nonempty string")

        tokens = self.values.get("max_tokens")
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("Codex max_tokens must be a positive integer")


    def wrapper_options(self, output_dir: Path) -> dict[str, Any]:
        """Build OPAL options without importing or launching an agent.

        Args:
            output_dir (Path): Directory for the experiment's optimizer records.

        Returns:
            dict[str, Any]: Fixed execution options and validated agent settings.
        """
        options = {key: self.values[key] for key in ("backend", "goal", "decisions", "every")}
        options.update(
            run_dir=str(output_dir / "opal"),
            max_decisions=self.values["decisions"],
            blocking=True,
            summarize=False,
            share_code=False,
        )
        if "schedule" in self.values:
            options["schedule"] = self.values["schedule"]
        if self.values["backend"] == "codex":
            options["backend_options"] = {
                key: self.values[key] for key in ("model", "effort", "max_tokens")
            } | {
                "timeout": 300,
                "unavailable_timeout": 60,
                "retries": 0,
                "binary": CodexRuntime.resolve_binary(self.values.get("binary")),
            }
        return options


class OptimizerAdapter:
    """Keep Adam and OPAL update calls behind one training interface."""

    def __init__(self, optimizer: Any, settings: AgentOptimizerSettings) -> None:
        """Initialize the optimizer adapter with the given optimizer and settings.

        Args:
            optimizer (Any): The underlying optimizer instance.
            settings (AgentOptimizerSettings): Validated optimizer-agent settings.
        """
        self.optimizer = optimizer
        self.settings = settings
        self.updates = 0


    @classmethod
    def create(
        cls,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        total_steps: int,
        output_dir: Path,
        settings: dict[str, Any] | None = None,
        allow_live_agent: bool = False,
    ) -> "OptimizerAdapter":
        """Wrap Adam only when the recipe explicitly enables OPAL.

        Args:
            optimizer (torch.optim.Optimizer): Starting optimizer for parameter updates.
            model (torch.nn.Module): Model whose parameters the optimizer updates.
            total_steps (int): Maximum optimizer updates in the experiment.
            output_dir (Path): Parent directory for OPAL records.
            settings (dict[str, Any] | None): Optional optimizer_agent recipe.
            allow_live_agent (bool): Explicit permission to construct the live backend.

        Returns:
            OptimizerAdapter: Adapter for the configured update mechanism.
        """
        validated = AgentOptimizerSettings(settings)
        if validated.values.get("schedule") and validated.values["schedule"][-1] >= total_steps:
            raise ValueError("Scheduled decisions must precede the maximum training update")
        if validated.values:
            if validated.values["backend"] == "codex" and not allow_live_agent:
                raise RuntimeError("Live OPAL requires explicit --allow-live-agent opt-in")
            options = validated.wrapper_options(output_dir)
            optimizer = import_module("opal").wrap(
                optimizer, model, total_steps=total_steps, **options
            )

        return cls(optimizer, validated)


    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear parameter gradients before the next backward pass.

        Args:
            set_to_none (bool): Use absent gradients rather than zero-filled tensors.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)


    def step(self, loss: torch.Tensor) -> None:
        """Apply one update and retain the actual update count.

        Args:
            loss (torch.Tensor): Current training loss passed only to OPAL.
        """
        if self.settings.values:
            self.optimizer.step(loss=loss)
        else:
            self.optimizer.step()
        self.updates += 1


    def close(self) -> None:
        """Finish OPAL records without calling any model summarizer."""
        if self.settings.values:
            self.optimizer.close()


    def provenance(self) -> dict[str, Any]:
        """Describe the optimizer path and actual update count.

        Returns:
            dict[str, Any]: Settings, package version, and completed updates.
        """
        record = {"base_optimizer": "Adam", "updates": self.updates}
        if self.settings.values:
            record.update(
                agent=self.settings.values,
                opal_version=version("opal-optimizer"),
                validation_shared_with_agent=False,
                blocking=True,
                summarize=False,
                share_code=False,
            )
        return record
