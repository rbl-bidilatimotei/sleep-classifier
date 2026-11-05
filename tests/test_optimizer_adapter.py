from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")

from sleep_classifier.optimizer_adapter import (  # noqa: E402
    AgentOptimizerSettings,
    OptimizerAdapter,
)


def _settings(backend="keep"):
    settings = {"backend": backend, "goal": "Reduce training loss", "decisions": 2, "every": 50}
    if backend == "codex":
        settings.update(model="gpt-6.1-sol", effort="high", max_tokens=20000)
    return settings


def test_plain_optimizer_never_imports_opal(tmp_path, monkeypatch):
    loader = Mock(side_effect=AssertionError("Unexpected import"))
    monkeypatch.setattr("sleep_classifier.optimizer_adapter.import_module", loader)
    optimizer = Mock()
    adapter = OptimizerAdapter.create(optimizer, Mock(), 10, tmp_path)
    loss = Mock()
    adapter.zero_grad()
    adapter.step(loss)
    adapter.close()
    optimizer.zero_grad.assert_called_once_with(set_to_none=True)
    optimizer.step.assert_called_once_with()
    optimizer.close.assert_not_called()
    assert adapter.provenance() == {"base_optimizer": "Adam", "updates": 1}


def test_keep_wraps_with_fixed_boundaries_and_closes(tmp_path, monkeypatch):
    wrapped = Mock()
    wrapper = Mock(return_value=wrapped)
    monkeypatch.setattr(
        "sleep_classifier.optimizer_adapter.import_module",
        Mock(return_value=SimpleNamespace(wrap=wrapper)),
    )
    adapter = OptimizerAdapter.create(Mock(), Mock(), 200, tmp_path, _settings())
    loss = Mock()
    adapter.step(loss)
    adapter.close()
    options = wrapper.call_args.kwargs
    assert options["backend"] == "keep"
    assert options["max_decisions"] == options["decisions"] == 2
    assert options["total_steps"] == 200
    assert options["blocking"] is True
    assert options["summarize"] is options["share_code"] is False
    assert "validation" not in options
    wrapped.step.assert_called_once_with(loss=loss)
    wrapped.close.assert_called_once_with()


def test_live_backend_requires_separate_opt_in(tmp_path, monkeypatch):
    loader = Mock(side_effect=AssertionError("Agent must not start"))
    monkeypatch.setattr("sleep_classifier.optimizer_adapter.import_module", loader)
    with pytest.raises(RuntimeError, match="allow-live-agent"):
        OptimizerAdapter.create(Mock(), Mock(), 100, tmp_path, _settings("codex"))
    loader.assert_not_called()


def test_live_backend_passes_explicit_soft_token_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sleep_classifier.optimizer_adapter.CodexRuntime.resolve_binary",
        Mock(return_value="/test/codex"),
    )
    wrapper = Mock()
    monkeypatch.setattr(
        "sleep_classifier.optimizer_adapter.import_module",
        Mock(return_value=SimpleNamespace(wrap=wrapper)),
    )
    OptimizerAdapter.create(Mock(), Mock(), 100, tmp_path, _settings("codex"), True)
    options = wrapper.call_args.kwargs["backend_options"]
    assert options == {
        "model": "gpt-6.1-sol",
        "effort": "high",
        "max_tokens": 20000,
        "timeout": 300,
        "unavailable_timeout": 60,
        "retries": 0,
        "binary": "/test/codex",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"backend": "claude"},
        {"goal": ""},
        {"decisions": 4},
        {"every": True},
        {"max_cost_usd": 5},
        {"max_tokens": None},
        {"effort": "low"},
    ],
)
def test_settings_reject_unbounded_or_unsupported_options(change):
    with pytest.raises(ValueError):
        AgentOptimizerSettings(_settings("codex") | change)


@pytest.mark.parametrize("settings", [[], "keep", 1])
def test_settings_reject_non_mapping_configuration(settings):
    with pytest.raises(ValueError, match="mapping"):
        AgentOptimizerSettings(settings)


@pytest.mark.parametrize("schedule", [[0], [50, 50], [51], [100, 50], [True]])
def test_settings_reject_invalid_explicit_schedule(schedule):
    settings = _settings() | {"decisions": len(schedule), "schedule": schedule}
    with pytest.raises(ValueError, match="schedule"):
        AgentOptimizerSettings(settings)


def test_early_schedule_is_forwarded_to_wrapper(tmp_path, monkeypatch):
    wrapper = Mock()
    monkeypatch.setattr(
        "sleep_classifier.optimizer_adapter.import_module",
        Mock(return_value=SimpleNamespace(wrap=wrapper)),
    )
    settings = _settings() | {"decisions": 1, "schedule": [50]}
    OptimizerAdapter.create(Mock(), Mock(), 6240, tmp_path, settings)
    assert wrapper.call_args.kwargs["schedule"] == [50]
    assert wrapper.call_args.kwargs["max_decisions"] == 1
    with pytest.raises(ValueError, match="precede"):
        OptimizerAdapter.create(Mock(), Mock(), 50, tmp_path, settings)
