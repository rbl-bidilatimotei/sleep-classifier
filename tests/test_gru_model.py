import pytest

torch = pytest.importorskip("torch")

from sleep_classifier.gru_model import CausalGRUClassifier  # noqa: E402


def test_classifier_returns_four_logits_for_each_sequence():
    model = CausalGRUClassifier(input_size=6, hidden_size=8, num_classes=4)
    features = torch.randn(3, 5, 6)
    lengths = torch.tensor([5, 3, 1], dtype=torch.int64)

    assert model(features, lengths).shape == (3, 4)


def test_classifier_ignores_padded_positions():
    model = CausalGRUClassifier()
    features = torch.randn(2, 4, 6)
    lengths = torch.tensor([2, 4], dtype=torch.int64)
    changed = features.clone()
    changed[0, 2:] = -500

    torch.testing.assert_close(model(features, lengths), model(changed, lengths))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_matches_cpu_ignores_padding_and_updates_parameters():
    torch.manual_seed(42)
    cpu_model = CausalGRUClassifier(hidden_size=8)
    mps_model = CausalGRUClassifier(hidden_size=8).to("mps")
    mps_model.load_state_dict(cpu_model.state_dict())
    features = torch.randn(3, 5, 6)
    lengths = torch.tensor([5, 1, 3], dtype=torch.int64)
    changed = features.clone()
    changed[1, 1:] = -500
    changed[2, 3:] = 500

    expected = cpu_model(features, lengths)
    actual = mps_model(features.to("mps"), lengths)
    padded = mps_model(changed.to("mps"), lengths)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual, padded)

    before = mps_model.classifier.weight.detach().clone()
    optimizer = torch.optim.Adam(mps_model.parameters(), lr=0.001)
    loss = torch.nn.functional.cross_entropy(actual, torch.tensor([0, 1, 2], device="mps"))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mps_model.parameters(), 1.0)
    optimizer.step()

    assert torch.isfinite(loss).item()
    assert all(torch.isfinite(parameter).all().item() for parameter in mps_model.parameters())
    assert not torch.equal(before, mps_model.classifier.weight)
