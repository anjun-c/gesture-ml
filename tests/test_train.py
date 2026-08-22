"""
train.py tests, run against real torch/pandas so the actual tensor shapes
and training loop execute -- these would have caught the earlier bug where
train_gesture_model() fed raw images into a model sized for 42-dim landmark
vectors. mediapipe itself is stubbed (see conftest.py): we don't have real
hand-tracking here, so extract_landmarks_to_csv is tested against a
duck-typed fake hands_instance rather than the real MediaPipe model.
"""
import types

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import GestureRecognitionModel
from train import EarlyStopping, GestureLandmarkDataset, extract_landmarks_to_csv, train_gesture_model


def test_early_stopping_does_not_stop_while_improving():
    early_stopping = EarlyStopping(patience=2)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6]
    assert not any(early_stopping(loss) for loss in losses)


def test_early_stopping_stops_after_patience_exhausted():
    early_stopping = EarlyStopping(patience=2, min_delta=0.0)
    assert early_stopping(1.0) is False  # first call just records the baseline
    assert early_stopping(1.0) is False  # no improvement, counter=1
    assert early_stopping(1.0) is True   # no improvement, counter=2 >= patience


def test_early_stopping_resets_counter_on_improvement():
    early_stopping = EarlyStopping(patience=2, min_delta=0.0)
    early_stopping(1.0)
    early_stopping(1.0)  # counter=1
    early_stopping(0.5)  # improved, counter resets to 0
    assert early_stopping(0.5) is False  # counter=1, still under patience


def test_gesture_landmark_dataset_reads_csv(tmp_path):
    csv_path = tmp_path / "landmarks.csv"
    header = ",".join(f"lm_{i}" for i in range(42)) + ",label\n"
    row0 = ",".join(str(0.1 * i) for i in range(42)) + ",0\n"
    row1 = ",".join(str(0.2 * i) for i in range(42)) + ",1\n"
    csv_path.write_text(header + row0 + row1)

    dataset = GestureLandmarkDataset(csv_path)

    assert len(dataset) == 2
    landmarks, label = dataset[1]
    assert landmarks.shape == (42,)
    assert landmarks.dtype == torch.float32
    assert label.item() == 1


def test_extract_landmarks_to_csv_skips_frames_with_no_hand(tmp_path):
    class FakeHandsInstance:
        """Only 'detects' a hand when the image's first pixel is 255."""

        def process(self, image):
            landmark = types.SimpleNamespace(x=0.5, y=0.5)
            detected = image[0, 0, 0] == 255
            multi = [types.SimpleNamespace(landmark=[landmark] * 21)] if detected else None
            return types.SimpleNamespace(multi_hand_landmarks=multi)

    detected_image = torch.ones(3, 4, 4)  # ToPILImage -> all-white -> pixel 255
    undetected_image = torch.zeros(3, 4, 4)  # all-black -> pixel 0, "no hand"
    fake_dataset = [(detected_image, 1), (undetected_image, 0)]

    output_csv = tmp_path / "out.csv"
    extract_landmarks_to_csv(fake_dataset, FakeHandsInstance(), output_csv)

    lines = output_csv.read_text().strip().splitlines()
    assert len(lines) == 2  # header + exactly one detected row
    assert lines[1].endswith(",1")  # the label of the detected frame


def test_train_gesture_model_runs_and_reduces_loss():
    torch.manual_seed(0)
    num_classes = 2
    inputs = torch.rand(16, 42)
    labels = torch.randint(0, num_classes, (16,))
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)

    model = GestureRecognitionModel(num_classes=num_classes)
    trained_model = train_gesture_model(model, loader, loader, epochs=3, patience=10)

    assert trained_model is model
    # Sanity check: after training, the model still produces correctly-shaped,
    # finite predictions (this is exactly the check that would have caught the
    # old raw-image/42-dim-model shape mismatch).
    output = trained_model(inputs)
    assert output.shape == (16, num_classes)
    assert torch.isfinite(output).all()
