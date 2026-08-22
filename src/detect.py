"""
Real-time hand detection + gesture recognition + gesture-controlled media keys.

Requires a trained model checkpoint produced by train.py (default: gesture_model.pt
in the current directory). Run `python train.py` first.
"""
import time

import cv2
import mediapipe as mp
import numpy as np
from filterpy.kalman import KalmanFilter

from capture import initialize_webcam, display_frame, capture_video, release_resources
from model import DEFAULT_MODEL_PATH, load_model, gesture_recognition_integration
import win_control

# Map gesture class names (must match the folder/class names your dataset was
# trained on, see the `classes` saved by train.py) to an action. Gestures not
# present here are recognized but ignored. Edit these keys to match your dataset.
GESTURE_ACTIONS = {
    "palm": win_control.play_pause_media,
    "fist": win_control.mute_volume,
    "thumb": win_control.volume_up,
    "l": win_control.volume_down,
    "ok": win_control.next_track,
}

# Minimum time between two triggered actions, and how long a gesture must
# stay the same before it's treated as an intentional, stable input rather
# than single-frame jitter/noise.
ACTION_COOLDOWN_SECONDS = 1.0
STABLE_FRAMES_REQUIRED = 3


class GestureActionDebouncer:
    """Only fires an action once a gesture has been stable for several frames, and rate-limits repeats."""

    def __init__(self, cooldown_seconds=ACTION_COOLDOWN_SECONDS, stable_frames=STABLE_FRAMES_REQUIRED):
        self.cooldown_seconds = cooldown_seconds
        self.stable_frames = stable_frames
        self._last_gesture = None
        self._stable_count = 0
        self._last_fired_gesture = None
        self._last_fired_time = 0.0

    def update(self, gesture_name):
        """Feed the latest predicted gesture name (or None). Returns the action to call, or None."""
        if gesture_name == self._last_gesture:
            self._stable_count += 1
        else:
            self._last_gesture = gesture_name
            self._stable_count = 1

        if gesture_name is None or self._stable_count < self.stable_frames:
            return None

        action = GESTURE_ACTIONS.get(gesture_name)
        if action is None:
            return None

        now = time.monotonic()
        if gesture_name == self._last_fired_gesture and (now - self._last_fired_time) < self.cooldown_seconds:
            return None

        self._last_fired_gesture = gesture_name
        self._last_fired_time = now
        return action


def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])  # Initial state (x, y, x_velocity, y_velocity)
    kf.F = np.array([[1., 0., 1., 0.],
                      [0., 1., 0., 1.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])  # State transition matrix
    kf.H = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.]])  # Measurement matrix
    kf.P *= 1000.  # Initial covariance matrix
    kf.R = np.array([[5., 0.],
                      [0., 5.]])  # Measurement noise covariance
    return kf


def apply_kalman_filter(kf, hand_landmarks):
    """Apply Kalman Filter to hand landmarks for position smoothing."""
    if not hand_landmarks:
        return None, None

    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y

    kf.predict()
    kf.update(np.array([wrist_x, wrist_y]))

    return kf.x[0], kf.x[1]


def hand_detection_with_kalman(frame, hands, kalman_filters, gesture_model, gesture_classes, debouncer, mp_drawing, mp_hands_solution):
    """Detect hands, smooth position with a Kalman filter, recognize gestures, and trigger actions."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if idx < len(kalman_filters):
                filtered_x, filtered_y = apply_kalman_filter(kalman_filters[idx], hand_landmarks)
                if filtered_x is not None and filtered_y is not None:
                    h, w, _ = frame.shape
                    cv2.circle(frame, (int(filtered_x * w), int(filtered_y * h)), 10, (0, 255, 0), -1)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands_solution.HAND_CONNECTIONS)

            gesture_name = gesture_recognition_integration(hand_landmarks, gesture_model, gesture_classes)
            if gesture_name is not None:
                cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            action = debouncer.update(gesture_name)
            if action is not None:
                print(f"Triggering action for gesture: {gesture_name}")
                action()

    return frame


def main_hand_detection_optimized(model_path=DEFAULT_MODEL_PATH):
    """Main function: hand tracking + Kalman smoothing + gesture-controlled media keys."""
    try:
        gesture_model, gesture_classes = load_model(model_path)
    except FileNotFoundError:
        print(f"No trained model found at '{model_path}'. Run `python train.py` first.")
        return

    mp_hands_solution = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands_instance = mp_hands_solution.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    kalman_filters = [initialize_kalman_filter(), initialize_kalman_filter()]
    debouncer = GestureActionDebouncer()

    cap = initialize_webcam()
    if cap is None:
        return

    while cap.isOpened():
        frame = capture_video(cap)
        if frame is None:
            break

        frame_with_landmarks = hand_detection_with_kalman(
            frame, hands_instance, kalman_filters, gesture_model, gesture_classes, debouncer, mp_drawing, mp_hands_solution
        )
        display_frame(frame_with_landmarks)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_resources(cap)


if __name__ == "__main__":
    main_hand_detection_optimized()
