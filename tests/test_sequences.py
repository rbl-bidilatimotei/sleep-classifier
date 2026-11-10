import pandas as pd
import pytest

torch = pytest.importorskip("torch")
StandardScaler = pytest.importorskip("sklearn.preprocessing").StandardScaler

from sleep_classifier.features import FEATURE_COLUMNS  # noqa: E402
from sleep_classifier.sequences import CausalSequenceDataset, SequenceBatchCollator  # noqa: E402


def make_frame():
    rows = []
    for participant, base in (("p1", 0), ("p2", 1000)):
        for index in range(3):
            row = {column: float(base + index) for column in FEATURE_COLUMNS}
            row.update(
                participant_id=participant,
                segment_id=0,
                time=index * 30,
                target=index,
                eligible=True,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def scaler_for(frame):
    return StandardScaler().fit(frame.loc[:, FEATURE_COLUMNS])


class TransformSpy:
    def __init__(self):
        self.calls = 0

    def transform(self, features):
        self.calls += 1
        return features.to_numpy(dtype=float)


def test_windows_use_only_past_rows_and_warm_up_at_every_epoch():
    frame = make_frame()
    scaler = scaler_for(frame)
    dataset = CausalSequenceDataset(frame.iloc[:3], scaler, context_epochs=2)
    extended = pd.concat([frame.iloc[:3], frame.iloc[[3]].assign(time=90)], ignore_index=True)
    extended_dataset = CausalSequenceDataset(extended, scaler, context_epochs=2)

    assert [len(dataset[i]["features"]) for i in range(3)] == [1, 2, 2]
    for index in range(3):
        torch.testing.assert_close(dataset[index]["features"], extended_dataset[index]["features"])


def test_scaler_transforms_the_ordered_frame_once():
    frame = make_frame()
    scaler = TransformSpy()

    CausalSequenceDataset(frame, scaler)

    assert scaler.calls == 1


def test_windows_never_cross_segments_or_participants():
    frame = make_frame()
    frame.loc[2, "segment_id"] = 1
    dataset = CausalSequenceDataset(frame, scaler_for(frame), context_epochs=10)

    assert [len(dataset[i]["features"]) for i in range(len(dataset))] == [1, 2, 1, 1, 2, 3]
    assert [dataset[i]["participant_id"] for i in range(len(dataset))] == ["p1"] * 3 + ["p2"] * 3


def test_invalid_gap_is_rejected():
    frame = make_frame()
    frame.loc[2, "time"] = 90

    with pytest.raises(ValueError, match="contiguous"):
        CausalSequenceDataset(frame, scaler_for(frame))


def test_collator_right_pads_and_model_ignores_padding():
    from sleep_classifier.gru_model import CausalGRUClassifier

    frame = make_frame().iloc[:3]
    dataset = CausalSequenceDataset(frame, scaler_for(frame), context_epochs=3)
    batch = SequenceBatchCollator()([dataset[0], dataset[2]])
    model = CausalGRUClassifier()
    logits = model(batch["features"], batch["lengths"])
    changed = batch["features"].clone()
    changed[0, 1:] = 10000

    torch.testing.assert_close(logits, model(changed, batch["lengths"]))
    assert batch["features"].shape == (2, 3, len(FEATURE_COLUMNS))
    assert batch["lengths"].dtype == torch.int64 and batch["lengths"].device.type == "cpu"
    assert batch["times"].dtype == torch.float64
    assert logits.shape == (2, 4)


def test_rejects_invalid_context_and_unscored_rows():
    frame = make_frame()
    for context in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            CausalSequenceDataset(frame, scaler_for(frame), context_epochs=context)
    frame.loc[0, "eligible"] = False
    with pytest.raises(ValueError, match="eligible"):
        CausalSequenceDataset(frame, scaler_for(frame))


def test_context_thirty_keeps_targets_and_restarts_after_gaps() -> None:
    """Retain every target while extending only participant-local contiguous history."""
    rows = []
    for participant in ("p1", "p2"):
        for index in range(45):
            row = {column: float(index) for column in FEATURE_COLUMNS}
            row.update(participant_id=participant, segment_id=int(index >= 35),
                       time=(index + (5 if index >= 35 else 0)) * 30,
                       target=index % 4, eligible=True)
            rows.append(row)
    frame = pd.DataFrame(rows)
    shorter = CausalSequenceDataset(frame, TransformSpy(), context_epochs=10)
    longer = CausalSequenceDataset(frame, TransformSpy(), context_epochs=30)

    assert len(shorter) == len(longer) == 90
    assert len(shorter[29]["features"]) == 10
    assert len(longer[29]["features"]) == 30
    assert len(longer[35]["features"]) == 1
    assert len(longer[45]["features"]) == 1
    for short, long in zip(shorter, longer):
        assert (short["participant_id"], short["time"], short["target"].item()) == (
            long["participant_id"], long["time"], long["target"].item(),
        )
        torch.testing.assert_close(short["features"][-1], long["features"][-1])
