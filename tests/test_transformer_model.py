from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from sleep_classifier.transformer_model import CausalTransformerClassifier  # noqa: E402


def test_transformer_returns_one_stage_vector_per_valid_sequence() -> None:
    """Return four stage logits for every padded batch item."""
    model = CausalTransformerClassifier(dropout=0).eval()
    features = torch.randn(3, 5, 6)
    lengths = torch.tensor([5, 3, 1])

    assert model(features, lengths).shape == (3, 4)


def test_transformer_ignores_padding_and_matches_unpadded_examples() -> None:
    """Preserve predictions when padding changes or each sequence runs separately."""
    model = CausalTransformerClassifier(dropout=0).eval()
    features = torch.randn(3, 5, 6)
    lengths = torch.tensor([5, 1, 3])
    changed = features.clone()
    changed[1, 1:] = 1000
    changed[2, 3:] = -1000

    with torch.no_grad():
        expected = model(features, lengths)
        actual = model(changed, lengths)
        individual = torch.cat([
            model(features[index:index + 1, :length], torch.tensor([length]))
            for index, length in enumerate(lengths.tolist())
        ])

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(individual, expected)


def test_encoder_representation_does_not_depend_on_later_epochs() -> None:
    """Keep earlier encoder representations unchanged after modifying later epochs."""
    model = CausalTransformerClassifier(dropout=0).eval()
    features = torch.randn(1, 6, 6)
    changed = features.clone()
    changed[:, 3:] = torch.randn(1, 3, 6) * 100
    capture = Mock(return_value=None)
    hook = model.encoder.register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(features, torch.tensor([6]))
            model(changed, torch.tensor([6]))
    finally:
        hook.remove()

    encoded = [call.args[2] for call in capture.call_args_list]
    torch.testing.assert_close(encoded[0][:, :3], encoded[1][:, :3])
    assert not torch.allclose(encoded[0][:, 3:], encoded[1][:, 3:])


def test_encoder_layers_do_not_start_with_identical_matrix_parameters() -> None:
    """Initialize stacked attention layers with independent matrix values."""
    model = CausalTransformerClassifier(num_layers=2)
    first = model.encoder.layers[0].self_attn.in_proj_weight
    second = model.encoder.layers[1].self_attn.in_proj_weight

    assert not torch.equal(first, second)
    assert first.data_ptr() != second.data_ptr()


@pytest.mark.parametrize("settings", [
    {"d_model": 15}, {"nhead": 0}, {"num_layers": True}, {"input_size": -1},
    {"feedforward_dim": 0}, {"max_context": 0}, {"num_classes": 0},
    {"dropout": -0.1}, {"dropout": 1}, {"dropout": float("nan")}, {"dropout": True},
])
def test_transformer_rejects_invalid_architecture_settings(settings: dict[str, object]) -> None:
    """Reject malformed model settings before constructing attention layers.

    Args:
        settings (dict[str, object]): Invalid constructor overrides.
    """
    with pytest.raises(ValueError):
        CausalTransformerClassifier(**settings)


@pytest.mark.parametrize("features,lengths", [
    (torch.randn(2, 5, 7), torch.tensor([5, 3])),
    (torch.randn(2, 11, 6), torch.tensor([5, 3])),
    (torch.randn(2, 5, 6), torch.tensor([0, 3])),
    (torch.randn(2, 5, 6), torch.tensor([6, 3])),
    (torch.randn(2, 5, 6), torch.tensor([5.0, 3.0])),
    (torch.randn(2, 5, 6), torch.tensor([3])),
    (torch.ones(2, 5, 6, dtype=torch.int64), torch.tensor([5, 3])),
])
def test_transformer_rejects_invalid_inputs(
    features: torch.Tensor, lengths: torch.Tensor
) -> None:
    """Reject invalid feature shapes, dtypes, and sequence lengths.

    Args:
        features (torch.Tensor): Invalid input batch or companion of invalid lengths.
        lengths (torch.Tensor): Supplied sequence lengths.
    """
    model = CausalTransformerClassifier()

    with pytest.raises(ValueError):
        model(features, lengths)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_forward_backward_matches_cpu_and_updates_parameters() -> None:
    """Verify MPS numerical agreement and a synthetic gradient update."""
    torch.manual_seed(42)
    cpu_model = CausalTransformerClassifier(dropout=0).eval()
    mps_model = CausalTransformerClassifier(dropout=0).to("mps").eval()
    mps_model.load_state_dict(cpu_model.state_dict())
    features = torch.randn(3, 5, 6)
    lengths = torch.tensor([5, 1, 3])

    expected = cpu_model(features, lengths)
    actual = mps_model(features.to("mps"), lengths)
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=1e-4)

    before = mps_model.input_projection.weight.detach().clone()
    optimizer = torch.optim.Adam(mps_model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(actual, torch.tensor([0, 1, 2], device="mps"))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mps_model.parameters(), 1.0)
    optimizer.step()

    assert torch.isfinite(loss).item()
    assert all(torch.isfinite(parameter).all().item() for parameter in mps_model.parameters())
    assert not torch.equal(before, mps_model.input_projection.weight)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_default_dropout_recipe_trains_a_full_padded_mps_batch() -> None:
    """Exercise the configured residual dropout on a full synthetic MPS batch."""
    model = CausalTransformerClassifier().to("mps").train()
    features = torch.randn(64, 10, 6, device="mps")
    lengths = torch.arange(64, dtype=torch.long) % 10 + 1
    targets = torch.arange(64, device="mps") % 4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.input_projection.weight.detach().clone()

    logits = model(features, lengths)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    assert logits.shape == (64, 4)
    assert torch.isfinite(loss).item()
    assert model.encoder.layers[0].dropout1.p == 0.1
    assert model.encoder.layers[0].dropout2.p == 0.1
    assert model.encoder.layers[0].dropout.p == 0.1
    assert not torch.equal(before, model.input_projection.weight)
    assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
