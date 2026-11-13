from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from sleep_classifier.patch_transformer_model import CausalPatchTransformerClassifier  # noqa: E402
from sleep_classifier.transformer_model import CausalTransformerClassifier  # noqa: E402


def test_patches_preserve_chronological_features_and_align_each_target_at_the_end() -> None:
    """Keep feature order and independently left-pad each sample's partial first patch."""
    model = CausalPatchTransformerClassifier(input_size=2, max_context=5, patch_size=3)
    features = torch.arange(1, 31, dtype=torch.float32).reshape(3, 5, 2)
    patches, lengths = model.patch_features(features, torch.tensor([5, 4, 1]))

    torch.testing.assert_close(patches, torch.tensor([
        [[0, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10]],
        [[0, 0, 0, 0, 11, 12], [13, 14, 15, 16, 17, 18]],
        [[0, 0, 0, 0, 21, 22], [0, 0, 0, 0, 0, 0]],
    ], dtype=torch.float32))
    assert lengths.tolist() == [2, 2, 1]


@pytest.mark.parametrize("length", range(1, 8))
def test_last_valid_epoch_always_occupies_the_final_patch_slot(length: int) -> None:
    """Keep the target at the same within-patch position for every warm-up length.

    Args:
        length (int): Number of completed raw epochs in the synthetic sequence.
    """
    model = CausalPatchTransformerClassifier(input_size=1, max_context=7, patch_size=3)
    features = torch.arange(1, 8, dtype=torch.float32).reshape(1, 7, 1)
    patches, counts = model.patch_features(features, torch.tensor([length]))
    valid = patches[0, :counts[0]].flatten()

    assert valid[-1] == length
    assert valid[valid > 0].tolist() == list(range(1, length + 1))
    assert counts.item() == (length + 2) // 3


def test_single_epoch_and_full_history_produce_one_and_ten_valid_tokens() -> None:
    """Handle a cold start and complete thirty-epoch history with one public interface."""
    model = CausalPatchTransformerClassifier(dropout=0.2).eval()
    features = torch.randn(2, 30, 6)
    patches, lengths = model.patch_features(features, torch.tensor([1, 30]))

    assert patches.shape == (2, 10, 18)
    assert lengths.tolist() == [1, 10]
    torch.testing.assert_close(patches[0, 0, :12], torch.zeros(12))
    torch.testing.assert_close(patches[0, 0, 12:], features[0, 0])
    assert model(features, torch.tensor([1, 30])).shape == (2, 4)
    assert model.backbone.max_context == 10
    assert sum(parameter.numel() for parameter in model.parameters()) == 2756


def test_padding_values_and_nan_do_not_change_individual_or_batched_predictions() -> None:
    """Ignore all raw padding and preserve predictions across supported batch shapes."""
    model = CausalPatchTransformerClassifier(max_context=9, dropout=0).eval()
    features = torch.randn(3, 8, 6)
    lengths = torch.tensor([8, 4, 1])
    changed = features.clone()
    changed[1, 4:] = float("nan")
    changed[2, 1:] = 1000
    with torch.no_grad():
        expected = model(features, lengths)
        actual = model(changed, lengths)
        individual = torch.cat([
            model(features[index:index + 1, :length], torch.tensor([length]))
            for index, length in enumerate(lengths.tolist())
        ])

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(individual, expected)


def test_appended_future_values_do_not_change_a_fixed_target_prediction() -> None:
    """Exclude appended epochs when the valid completed-history length stays fixed."""
    model = CausalPatchTransformerClassifier(max_context=9, dropout=0).eval()
    observed = torch.randn(1, 4, 6)
    appended = torch.cat([observed, torch.randn(1, 5, 6) * 1000], dim=1)
    with torch.no_grad():
        expected = model(observed, torch.tensor([4]))
        actual = model(appended, torch.tensor([4]))

    torch.testing.assert_close(actual, expected)


def test_earlier_patch_representation_excludes_later_completed_patches() -> None:
    """Preserve the causal encoder mask after jointly embedding each completed patch."""
    model = CausalPatchTransformerClassifier(max_context=6, dropout=0).eval()
    features = torch.randn(1, 6, 6)
    changed = features.clone()
    changed[:, 3:] += 100
    capture = Mock(return_value=None)
    hook = model.backbone.encoder.register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(features, torch.tensor([6]))
            model(changed, torch.tensor([6]))
    finally:
        hook.remove()
    outputs = [call.args[2] for call in capture.call_args_list]

    torch.testing.assert_close(outputs[0][:, :1], outputs[1][:, :1])
    assert not torch.allclose(outputs[0][:, 1:], outputs[1][:, 1:])


def test_model_preserves_batch_independence() -> None:
    """Prevent one participant's feature changes from affecting another batch item."""
    model = CausalPatchTransformerClassifier(max_context=6, dropout=0).eval()
    features = torch.randn(2, 6, 6)
    changed = features.clone()
    changed[1] += 100
    with torch.no_grad():
        expected = model(features, torch.tensor([6, 5]))
        actual = model(changed, torch.tensor([6, 5]))

    torch.testing.assert_close(actual[0], expected[0])


def test_patch_size_one_matches_the_existing_transformer() -> None:
    """Recover the reference computation when each token contains one raw epoch."""
    reference = CausalTransformerClassifier(max_context=7, dropout=0).eval()
    candidate = CausalPatchTransformerClassifier(max_context=7, dropout=0, patch_size=1).eval()
    candidate.backbone.load_state_dict(reference.state_dict())
    features = torch.randn(3, 7, 6)
    lengths = torch.tensor([7, 4, 1])
    with torch.no_grad():
        expected = reference(features, lengths)
        actual = candidate(features, lengths)

    torch.testing.assert_close(actual, expected)


def test_valid_inputs_and_weights_receive_gradients_but_padding_does_not() -> None:
    """Keep finite training gradients while excluding all raw padded observations."""
    model = CausalPatchTransformerClassifier(max_context=6, dropout=0)
    features = torch.randn(2, 6, 6)
    features[1, 2:] = float("nan")
    features.requires_grad_()
    loss = model(features, torch.tensor([6, 2])).square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()
    assert features.grad[0].abs().sum() > 0
    assert features.grad[1, :2].abs().sum() > 0
    assert torch.count_nonzero(features.grad[1, 2:]) == 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("settings", [
    {"patch_size": 0}, {"patch_size": True}, {"patch_size": 1.5}, {"patch_size": 31},
    {"input_size": 0}, {"input_size": True}, {"max_context": 0}, {"max_context": 1.5},
    {"d_model": 15}, {"nhead": 0}, {"num_layers": True}, {"feedforward_dim": 0},
    {"num_classes": 0}, {"dropout": -0.1}, {"dropout": 1.0}, {"dropout": float("nan")},
])
def test_patch_model_rejects_invalid_dimensions_and_dropout(settings: dict[str, object]) -> None:
    """Reject invalid patch and backbone controls before a model forward pass.

    Args:
        settings (dict[str, object]): Invalid constructor overrides.
    """
    with pytest.raises(ValueError):
        CausalPatchTransformerClassifier(**settings)


@pytest.mark.parametrize("shape,lengths,integer_features", [
    ((0, 5, 6), [], False), ((1, 0, 6), [1], False), ((1, 31, 6), [31], False),
    ((1, 4, 5), [4], False), ((1, 4), [4], False), ((2, 4, 6), [4], False),
    ((1, 4, 6), [0], False), ((1, 4, 6), [5], False), ((1, 4, 6), [1.5], False),
    ((1, 4, 6), [True], False), ((1, 4, 6), [4], True),
])
def test_patch_model_rejects_invalid_raw_inputs(
    shape: tuple[int, ...], lengths: list[int | float | bool], integer_features: bool,
) -> None:
    """Reject malformed raw batches and unsupported valid-length declarations.

    Args:
        shape (tuple[int, ...]): Raw input dimensions.
        lengths (list[int | float | bool]): Proposed valid raw epoch counts.
        integer_features (bool): Supply invalid integer features instead of floating values.
    """
    model = CausalPatchTransformerClassifier()
    features = torch.ones(shape, dtype=torch.long if integer_features else torch.float32)
    with pytest.raises(ValueError):
        model(features, torch.tensor(lengths))
