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
