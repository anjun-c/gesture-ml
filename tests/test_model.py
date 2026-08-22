"""
model.py tests, run against the real torch/numpy — these check the actual
tensor math and checkpoint round trip, not mocks.
"""
import types

import torch

from model import (
    GestureRecognitionModel,
    gesture_recognition_integration,
    load_model,
    recognize_gesture,
)


def _fake_hand_landmarks(xs, ys):
    """Build a MediaPipe-shaped hand_landmarks stand-in from parallel x/y lists."""
    landmark = [types.SimpleNamespace(x=x, y=y) for x, y in zip(xs, ys)]
    return types.SimpleNamespace(landmark=landmark)


def test_model_forward_pass_shape():
    model = GestureRecognitionModel(num_classes=5)
    model.eval()
    batch = torch.rand(3, 42)
    output = model(batch)
    assert output.shape == (3, 5)


def test_load_model_round_trip(tmp_path):
    classes = ["fist", "palm", "ok"]
    model = GestureRecognitionModel(num_classes=len(classes))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "classes": classes}, checkpoint_path)

    loaded_model, loaded_classes = load_model(checkpoint_path)

    assert loaded_classes == classes
    assert loaded_model.training is False  # load_model must leave it in eval mode
    output = loaded_model(torch.rand(1, 42))
    assert output.shape == (1, len(classes))


def test_recognize_gesture_returns_valid_class_index():
    model = GestureRecognitionModel(num_classes=4)
    model.eval()
    predicted = recognize_gesture(model, [0.1] * 42)
    assert isinstance(predicted, int)
    assert 0 <= predicted < 4


def test_gesture_recognition_integration_returns_none_without_hand():
    model = GestureRecognitionModel(num_classes=4)
    assert gesture_recognition_integration(None, model, ["a", "b", "c", "d"]) is None


def test_gesture_recognition_integration_returns_class_name():
    classes = ["fist", "palm"]
    model = GestureRecognitionModel(num_classes=len(classes))
    xs = [0.1 * i for i in range(21)]
    ys = [0.2 * i for i in range(21)]
    hand_landmarks = _fake_hand_landmarks(xs, ys)

    result = gesture_recognition_integration(hand_landmarks, model, classes)

    assert result in classes


def test_gesture_recognition_integration_uses_x_then_y_feature_order(monkeypatch):
    """
    Regression test: inference must build the landmark vector the same way
    training data was extracted (all x's, then all y's) -- an earlier
    version of this code flattened landmarks as interleaved [x0,y0,x1,y1,...],
    which silently mismatched the training feature order.
    """
    xs = [float(i) for i in range(21)]
    ys = [float(100 + i) for i in range(21)]
    hand_landmarks = _fake_hand_landmarks(xs, ys)

    captured = {}

    def fake_recognize_gesture(_model, landmarks):
        captured["landmarks"] = landmarks
        return 0

    monkeypatch.setattr("model.recognize_gesture", fake_recognize_gesture)

    gesture_recognition_integration(hand_landmarks, model=object(), classes=["only"])

    assert captured["landmarks"] == xs + ys
