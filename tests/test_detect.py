"""
detect.py tests. cv2's real color-conversion/drawing calls run for real
(no camera needed for those); mediapipe hand detection and win32 key events
are stubbed (see conftest.py) since neither can run in this environment --
mediapipe needs a real trained model + real images, and win32api needs an
actual Windows machine. These tests verify the wiring: a stable, mapped
gesture prediction results in exactly one call to the right action,
respecting the stability and cooldown debounce rules.
"""
import types
from unittest.mock import Mock

import numpy as np

import detect
from detect import GestureActionDebouncer, hand_detection_with_kalman, initialize_kalman_filter


def test_debouncer_requires_stable_frames_before_firing(monkeypatch):
    monkeypatch.setattr(detect, "GESTURE_ACTIONS", {"wave": Mock()})
    debouncer = GestureActionDebouncer(cooldown_seconds=0, stable_frames=3)
    assert debouncer.update("wave") is None  # count=1
    assert debouncer.update("wave") is None  # count=2, still below stable_frames=3
    assert debouncer.update("wave") is not None  # count=3, fires


def test_debouncer_fires_mapped_action_after_stable_frames(monkeypatch):
    mock_action = Mock()
    monkeypatch.setattr(detect, "GESTURE_ACTIONS", {"wave": mock_action})
    debouncer = GestureActionDebouncer(cooldown_seconds=0, stable_frames=2)

    assert debouncer.update("wave") is None
    fired = debouncer.update("wave")

    assert fired is mock_action


def test_debouncer_ignores_unmapped_gesture(monkeypatch):
    monkeypatch.setattr(detect, "GESTURE_ACTIONS", {"wave": Mock()})
    debouncer = GestureActionDebouncer(cooldown_seconds=0, stable_frames=1)
    assert debouncer.update("unknown_gesture") is None


def test_debouncer_ignores_none_gesture(monkeypatch):
    monkeypatch.setattr(detect, "GESTURE_ACTIONS", {"wave": Mock()})
    debouncer = GestureActionDebouncer(cooldown_seconds=0, stable_frames=1)
    assert debouncer.update(None) is None


def test_debouncer_switching_gesture_resets_stability(monkeypatch):
    monkeypatch.setattr(detect, "GESTURE_ACTIONS", {"wave": Mock(), "fist": Mock()})
    debouncer = GestureActionDebouncer(cooldown_seconds=0, stable_frames=2)

    assert debouncer.update("wave") is None  # count=1
    assert debouncer.update("fist") is None  # gesture changed, count resets to 1
    assert debouncer.update("fist") is not None  # count=2, fires


def test_debouncer_respects_cooldown_between_repeats(monkeypatch):
    mock_action = Mock()
    monkeypatch.setattr(detect, "GESTURE_ACTIONS", {"wave": mock_action})
    debouncer = GestureActionDebouncer(cooldown_seconds=100, stable_frames=1)

    first = debouncer.update("wave")
    second = debouncer.update("wave")  # still stable and mapped, but within cooldown

    assert first is mock_action
    assert second is None


def test_debouncer_fires_again_after_cooldown_elapses(monkeypatch):
    mock_action = Mock()
    monkeypatch.setattr(detect, "GESTURE_ACTIONS", {"wave": mock_action})
    debouncer = GestureActionDebouncer(cooldown_seconds=10, stable_frames=1)

    times = iter([0.0, 0.0, 11.0])
    monkeypatch.setattr(detect.time, "monotonic", lambda: next(times))

    assert debouncer.update("wave") is mock_action  # t=0, fires
    assert debouncer.update("wave") is None  # t=0 still, within cooldown
    assert debouncer.update("wave") is mock_action  # t=11, cooldown elapsed


def _fake_hand(xs, ys):
    landmark = [types.SimpleNamespace(x=x, y=y) for x, y in zip(xs, ys)]
    return types.SimpleNamespace(landmark=landmark)


def test_hand_detection_with_kalman_wires_recognized_gesture_to_action(monkeypatch):
    """
    End-to-end wiring test (no real camera/model): a hand is "detected" for
    several frames, gesture recognition is faked to always return 'wave',
    and after enough stable frames the mapped action must fire exactly once.
    """
    mock_action = Mock()
    monkeypatch.setattr(detect, "GESTURE_ACTIONS", {"wave": mock_action})
    monkeypatch.setattr(detect, "gesture_recognition_integration", lambda *_a, **_k: "wave")

    fake_hand = _fake_hand([0.5] * 21, [0.5] * 21)
    fake_hands = Mock()
    fake_hands.process.return_value = types.SimpleNamespace(multi_hand_landmarks=[fake_hand])

    fake_mp_drawing = Mock()
    fake_mp_hands_solution = types.SimpleNamespace(HAND_CONNECTIONS=[])

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    kalman_filters = [initialize_kalman_filter()]
    debouncer = GestureActionDebouncer(cooldown_seconds=0, stable_frames=2)

    for _ in range(2):
        hand_detection_with_kalman(
            frame, fake_hands, kalman_filters, gesture_model=None, gesture_classes=["wave"],
            debouncer=debouncer, mp_drawing=fake_mp_drawing, mp_hands_solution=fake_mp_hands_solution,
        )

    mock_action.assert_called_once()
