import copy
from typing import Any

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.window_feature_settings import WindowFeatureSettings  # noqa: E402
from sleep_classifier.window_heart_rate import WindowLocalHeartRate  # noqa: E402


def _settings(history: int = 10, minimum: int = 4) -> dict[str, Any]:
    """Construct the declared causal feature settings.

    Args:
        history (int): Maximum previous epochs used by the mean.
        minimum (int): Previous epochs required for availability.
    """
    return {"name": "heart_rate_deviation", "history_epochs": history,
            "min_history_epochs": minimum}


def _state(mean: float = 1.0, scale: float = 2.0) -> dict[str, Any]:
    """Create valid fitted metadata with explicit BPM normalization.

    Args:
        mean (float): Training deviation mean in BPM.
        scale (float): Training deviation population scale in BPM.
    """
    return {"schema_version": 1, "heart_rate_scale": 2.0, "delta_mean": mean,
            "delta_scale": scale, "available_training_targets": 100,
            "scaling_fit": "training_targets_only"}


def _window(heart: list[float]) -> torch.Tensor:
    """Create a six-channel raw window with only the heart-rate mean changing.

    Args:
        heart (list[float]): Training-scaled heart-rate means in chronological order.
    """
    result = torch.zeros(len(heart), len(FEATURE_COLUMNS))
    result[:, 0] = torch.tensor(heart)
    return result


def test_window_settings_keep_default_path_disabled_and_feature_order() -> None:
    """Keep raw names unchanged unless the explicit feature family is enabled."""
    raw = list(FEATURE_COLUMNS)
    disabled = WindowFeatureSettings(None)
    assert not disabled.enabled and disabled.values is None
    assert disabled.model_features(raw) == raw
    assert WindowFeatureSettings(_settings()).model_features(raw) == [
        *raw, "heart_rate_delta_past", "heart_history_available",
    ]
    assert raw == list(FEATURE_COLUMNS)


def test_legacy_settings_preserve_exact_metadata_state_and_random_stream() -> None:
    """Keep legacy constructors, buffers, normalization, and random state unchanged."""
    settings = _settings()
    configured = WindowFeatureSettings(settings)
    assert configured.values == settings
    assert set(configured.values) == {"name", "history_epochs", "min_history_epochs"}
    assert configured.masked_channels == WindowFeatureSettings(None).masked_channels == []
    before = torch.get_rng_state().clone()
    augment = WindowLocalHeartRate(settings, _state())
    assert torch.equal(torch.get_rng_state(), before)
    assert augment.settings == settings
    assert augment.normalization_state == _state()
    assert list(augment.parameters()) == []
    assert set(augment.state_dict()) == {"heart_rate_scale", "delta_mean", "delta_scale"}
    for name in augment.state_dict():
        assert torch.equal(augment.state_dict()[name], torch.tensor(_state()[name]))


@pytest.mark.parametrize("masked_channels", [
    ["heart_rate_delta_past"], ["heart_history_available"],
    ["heart_rate_delta_past", "heart_history_available"],
])
def test_mask_settings_copy_lists_and_keep_effective_channel_names(
    masked_channels: list[str],
) -> None:
    """Validate each supported mask without sharing mutable lists or removing channels.

    Args:
        masked_channels (list[str]): Supported derived channels to zero.
    """
    supplied = list(masked_channels)
    configured = WindowFeatureSettings(_settings() | {"masked_channels": supplied})
    supplied.clear()
    copied = configured.masked_channels
    copied.clear()
    assert configured.masked_channels == masked_channels
    assert configured.values == _settings() | {"masked_channels": masked_channels}
    assert configured.model_features(list(FEATURE_COLUMNS)) == [
        *FEATURE_COLUMNS, "heart_rate_delta_past", "heart_history_available",
    ]


@pytest.mark.parametrize("masked_channels", [
    None, [], (), ("heart_rate_delta_past",), "heart_rate_delta_past", 1, True,
    ["unknown"], ["heart_rate_mean"], ["heart_rate_delta_past", "heart_rate_delta_past"],
    [1], [["heart_rate_delta_past"]], {"heart_rate_delta_past": True},
])
def test_invalid_mask_settings_are_rejected(masked_channels: Any) -> None:
    """Reject empty, duplicate, unsupported, or nonlist masks before model construction.

    Args:
        masked_channels (Any): Invalid proposed channel selection.
    """
    with pytest.raises(ValueError, match="masked_channels"):
        WindowFeatureSettings(_settings() | {"masked_channels": masked_channels})


@pytest.mark.parametrize("masked_channels", [
    ["heart_rate_delta_past"], ["heart_history_available"],
    ["heart_rate_delta_past", "heart_history_available"],
])
def test_selected_channels_are_exactly_zero_after_scaling(
    masked_channels: list[str],
) -> None:
    """Zero only selected derived channels while retaining raw inputs and fitted metadata.

    Args:
        masked_channels (list[str]): Valid derived channels to zero after normalization.
    """
    inputs = _window([0.0, 1.0, 2.0, 5.0, 9.0]).unsqueeze(0)
    lengths = torch.tensor([5])
    settings = _settings(history=3, minimum=2)
    full = WindowLocalHeartRate(settings, _state())(inputs, lengths)
    masked = WindowLocalHeartRate(settings | {"masked_channels": masked_channels}, _state())
    actual = masked(inputs, lengths)
    assert actual.shape == (1, 5, 8)
    assert torch.equal(actual[:, :, :6], full[:, :, :6])
    for index, name in enumerate(("heart_rate_delta_past", "heart_history_available"), start=6):
        expected = (
            torch.zeros_like(full[:, :, index]) if name in masked_channels else full[:, :, index]
        )
        assert torch.equal(actual[:, :, index], expected)
    assert masked.normalization_state == _state()


def test_delta_mask_blocks_only_derived_delta_gradients() -> None:
    """Block deviation gradients while preserving raw-channel gradients through concatenation."""
    settings = _settings(history=3, minimum=2)
    full = WindowLocalHeartRate(settings, _state())
    masked = WindowLocalHeartRate(
        settings | {"masked_channels": ["heart_rate_delta_past"]}, _state(),
    )
    full_input = _window([0.0, 1.0, 2.0, 5.0, 9.0]).unsqueeze(0).requires_grad_()
    masked_input = full_input.detach().clone().requires_grad_()
    full(full_input, torch.tensor([5]))[:, :, 6].sum().backward()
    output = masked(masked_input, torch.tensor([5]))
    output[:, :, 6].sum().backward(retain_graph=True)
    assert full_input.grad is not None and torch.count_nonzero(full_input.grad) > 0
    assert torch.equal(masked_input.grad, torch.zeros_like(masked_input))
    output[:, :, :6].sum().backward()
    assert torch.equal(masked_input.grad, torch.ones_like(masked_input))


@pytest.mark.parametrize("masked_channels", [
    ["heart_rate_delta_past"], ["heart_history_available"],
    ["heart_rate_delta_past", "heart_history_available"],
])
def test_masks_preserve_padding_and_future_invariance(masked_channels: list[str]) -> None:
    """Keep padded output zero and earlier features independent of later observations.

    Args:
        masked_channels (list[str]): Derived channel mask used for both windows.
    """
    augment = WindowLocalHeartRate(_settings() | {"masked_channels": masked_channels}, _state())
    inputs = torch.stack([_window([float(x) for x in range(30)])] * 2)
    inputs[1, 7:, :] = float("nan")
    lengths = torch.tensor([30, 7])
    actual = augment(inputs, lengths)
    changed = inputs.clone()
    changed[0, 20:, :] = 10000
    changed[1, 7:, :] = 1e20
    after = augment(changed, lengths)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[1, 7:], torch.zeros(23, 8))
    assert torch.equal(actual[0, :20], after[0, :20])
    assert torch.equal(actual[1], after[1])


@pytest.mark.parametrize("masked_channels", [
    ["heart_rate_delta_past"], ["heart_history_available"],
    ["heart_rate_delta_past", "heart_history_available"],
])
def test_masking_preserves_training_normalization_fit(masked_channels: list[str]) -> None:
    """Fit the same available training deviations even when their model channels are masked.

    Args:
        masked_channels (list[str]): Derived channels selected for the later forward mask.
    """
    samples = [{"features": _window([0.0, 1.0, 3.0])},
               {"features": _window([0.0, 1.0, 3.0, 4.0])}]
    settings = _settings(history=3, minimum=2)
    expected = WindowLocalHeartRate.fit(samples, 2.0, settings)
    actual = WindowLocalHeartRate.fit(samples, 2.0, settings | {"masked_channels": masked_channels})
    assert actual == expected


def test_manual_feature_formula_and_startup_missingness() -> None:
    """Exclude the current epoch from history and impute only unavailable standardized deltas."""
    augment = WindowLocalHeartRate(_settings(history=3, minimum=2), _state())
    inputs = _window([0.0, 1.0, 2.0, 5.0, 9.0]).unsqueeze(0)
    before = inputs.clone()
    actual = augment(inputs, torch.tensor([5]))
    expected = torch.tensor([0.0, 0.0, 1.0, 3.5, (2 * (9 - 8 / 3) - 1) / 2])
    torch.testing.assert_close(actual[0, :, 6], expected)
    assert torch.equal(actual[0, :, 7], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))
    assert torch.equal(actual[:, :, :6], before)
    assert torch.equal(inputs, before)


def test_future_perturbation_cannot_change_earlier_features() -> None:
    """Keep prior derived channels invariant when a later observation changes."""
    augment = WindowLocalHeartRate(_settings(), _state())
    source = _window([float(x) for x in range(30)]).unsqueeze(0)
    changed = source.clone()
    changed[:, 20:, :] = 10000
    before = augment(source, torch.tensor([30]))
    after = augment(changed, torch.tensor([30]))
    assert torch.equal(before[:, :20], after[:, :20])


def test_window_local_boundary_does_not_reuse_an_older_prefix() -> None:
    """Reset derived history at the supplied window even when older observations exist elsewhere."""
    augment = WindowLocalHeartRate(_settings(), _state())
    full = _window([float(x) for x in range(40)])
    actual = augment(full[-30:].unsqueeze(0), torch.tensor([30]))
    altered = full.clone()
    altered[:10, 0] = 1e6
    same = augment(altered[-30:].unsqueeze(0), torch.tensor([30]))
    assert torch.equal(actual, same)
    assert torch.equal(actual[0, :4, 6:], torch.zeros(4, 2))
    assert actual[0, 4, 7] == 1.0
    session_computed = augment(full.unsqueeze(0), torch.tensor([40]))[:, -30:]
    assert not torch.equal(actual[:, :4, 6:], session_computed[:, :4, 6:])


def test_short_sequences_and_padding_cannot_supply_history() -> None:
    """Keep padded and short-history positions unavailable even with nonfinite padding."""
    augment = WindowLocalHeartRate(_settings(), _state())
    batch = torch.stack([_window([1.0] * 8), _window([2.0] * 8)])
    batch[0, 3:, :] = float("nan")
    batch[1, 6:, :] = 1e20
    result = augment(batch, torch.tensor([3, 6]))
    assert torch.isfinite(result).all()
    assert torch.equal(result[0, :, 6:], torch.zeros(8, 2))
    assert torch.equal(result[0, 3:], torch.zeros(5, 8))
    assert torch.equal(result[1, 6:], torch.zeros(2, 8))
    assert torch.equal(result[1, :, 7], torch.tensor([0, 0, 0, 0, 1, 1, 0, 0]))


def test_fit_uses_each_available_training_target_once_and_population_scale() -> None:
    """Fit from target deviations without weighting the overlapping token occurrences again."""
    samples = [{"features": _window([0.0, 1.0, 3.0])},
               {"features": _window([0.0, 1.0, 3.0, 4.0])},
               {"features": _window([100.0])}]
    before = copy.deepcopy(samples)
    result = WindowLocalHeartRate.fit(samples, 2.0, _settings(3, 2))
    # Raw target deviations are five and sixteen-thirds BPM
    expected_values = torch.tensor([5.0, 16 / 3], dtype=torch.float64)
    assert result["available_training_targets"] == 2
    assert result["delta_mean"] == pytest.approx(float(expected_values.mean()))
    assert result["delta_scale"] == pytest.approx(float(expected_values.std(correction=0)))
    assert result["scaling_fit"] == "training_targets_only"
    assert all(torch.equal(a["features"], b["features"])
               for a, b in zip(samples, before, strict=True))


def test_fit_zero_variance_and_unavailable_targets() -> None:
    """Use unit scale for a constant training delta and reject absent usable history."""
    samples = [{"features": _window([1.0] * 6)} for _ in range(3)]
    result = WindowLocalHeartRate.fit(samples, 7.0, _settings())
    assert result["delta_mean"] == 0.0 and result["delta_scale"] == 1.0
    with pytest.raises(ValueError, match="No training targets"):
        WindowLocalHeartRate.fit([{"features": _window([1.0] * 4)}], 7.0, _settings())


def test_normalization_state_is_copied_and_gradients_remain_finite() -> None:
    """Preserve provenance ownership and allow causal feature gradients to reach valid inputs."""
    state = _state()
    augment = WindowLocalHeartRate(_settings(), state)
    state["delta_mean"] = 999
    copied = augment.normalization_state
    copied["delta_mean"] = -999
    assert augment.normalization_state["delta_mean"] == 1.0
    raw = torch.randn(2, 12, 6, requires_grad=True)
    augmented = augment(raw, torch.tensor([12, 7]))
    augmented.square().mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert torch.equal(raw.grad[1, 7:], torch.zeros(5, 6))


@pytest.mark.parametrize("changes", [
    {"schema_version": True}, {"schema_version": 2}, {"heart_rate_scale": 0.0},
    {"delta_mean": float("nan")}, {"delta_scale": -1.0},
    {"available_training_targets": 0}, {"available_training_targets": True},
    {"scaling_fit": "validation"}, {"unused": 0},
])
def test_invalid_normalization_is_rejected(changes: dict[str, Any]) -> None:
    """Reject unsupported fitted state before inference.

    Args:
        changes (dict[str, Any]): Invalid normalization-field replacements.
    """
    with pytest.raises(ValueError):
        WindowLocalHeartRate(_settings(), _state() | changes)


@pytest.mark.parametrize("settings", [
    {}, {"name": "other", "history_epochs": 10, "min_history_epochs": 4},
    {"name": "heart_rate_deviation", "history_epochs": True, "min_history_epochs": 4},
    {"name": "heart_rate_deviation", "history_epochs": 10, "min_history_epochs": 0},
    {"name": "heart_rate_deviation", "history_epochs": 2, "min_history_epochs": 4},
])
def test_invalid_feature_settings_are_rejected(settings: dict[str, Any]) -> None:
    """Reject unused fields and ambiguous history bounds.

    Args:
        settings (dict[str, Any]): Invalid proposed window configuration.
    """
    with pytest.raises(ValueError):
        WindowFeatureSettings(settings)


def test_window_history_must_fit_inside_the_raw_context() -> None:
    """Reject a feature definition that would need data beyond the supplied raw context."""
    settings = WindowFeatureSettings(_settings())
    settings.validate_context(30)
    with pytest.raises(ValueError, match="shorter"):
        settings.validate_context(10)


@pytest.mark.parametrize("lengths", [torch.tensor([0]), torch.tensor([7]),
                                     torch.tensor([2.0]), torch.tensor([True])])
def test_invalid_lengths_are_rejected(lengths: torch.Tensor) -> None:
    """Reject lengths that would make validity or past counts ambiguous.

    Args:
        lengths (torch.Tensor): Invalid sequence length vector.
    """
    with pytest.raises(ValueError, match="lengths"):
        WindowLocalHeartRate(_settings(), _state())(torch.zeros(1, 6, 6), lengths)
