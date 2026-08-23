# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Webcam hand-gesture recognition: MediaPipe extracts hand landmarks, a small PyTorch MLP
classifies the gesture, and recognized gestures trigger Windows media-key actions
(play/pause, volume, track skip) via `win32api`. It's a personal "for fun" project, not
a package — there's no `setup.py`/build step, just scripts run directly from `src/`.

## Commands

Install:
```
pip install -r requirements-dev.txt   # adds pytest on top of requirements.txt
```

Run tests (from repo root — `pytest.ini` sets `testpaths = tests`):
```
pytest                                 # full suite
pytest tests/test_model.py             # one file
pytest tests/test_detect.py::test_debouncer_fires_mapped_action_after_stable_frames  # one test
```

Train a model (from `src/`, needs a real image dataset the repo doesn't ship — see README):
```
cd src && python train.py --data-dir ../data/archive
```
Writes `gesture_model.pt` (model weights + the class-name list, index-aligned with the
model's output logits) into the current directory.

Run live detection (Windows only — needs `gesture_model.pt` present and a webcam):
```
cd src && python detect.py
```

There is no linter/formatter configured; `python -m pyflakes src tests` is a reasonable
ad hoc check before committing.

## Architecture

### Train/inference split

Training and inference are deliberately separate modules so that importing the inference
path never re-triggers dataset loading or training:

- **`model.py`** — import-safe: the `GestureRecognitionModel` (a 42→128→64→num_classes MLP;
  input is 21 hand landmarks × (x, y)), `extract_landmarks_from_image` (MediaPipe → flat
  landmark list), and inference helpers `load_model()` / `recognize_gesture()` /
  `gesture_recognition_integration()`. No top-level side effects.
- **`train.py`** — everything data/training-related, guarded by `if __name__ == "__main__"`:
  `GestureLandmarkDataset` (reads a landmarks CSV), `EarlyStopping`, landmark extraction
  from an `ImageFolder` dataset to CSV, the training loop, and saving the checkpoint
  (`{"model_state_dict": ..., "classes": ...}`) via `torch.save`.
- **`detect.py`** — the live loop: webcam capture → MediaPipe hand detection → Kalman-smoothed
  wrist position (visual only) → `model.load_model()` (loaded once at startup, not retrained)
  → gesture prediction → debounced action dispatch.

**Landmark feature order matters and must stay consistent between training and inference:**
both `extract_landmarks_from_image` (training) and `gesture_recognition_integration`
(inference) build the 42-value vector as all 21 x-coordinates followed by all 21
y-coordinates (`[x0..x20, y0..y20]`), not interleaved. A prior version of this code had
these two paths mismatched, which silently produced wrong predictions (see HISTORY.md).
When touching either function, keep the ordering identical.

`num_classes` is never hardcoded — it's derived from `len(classes)`, where `classes` comes
from `ImageFolder.classes` at training time and is carried inside the checkpoint, so
`detect.py` doesn't need the dataset present to know the model's output layout.

### Gesture → action wiring (`detect.py`)

`GESTURE_ACTIONS` is a `dict[gesture_class_name, win_control_function]`. Its keys are
placeholders and must be edited to match whatever class/folder names the actual training
dataset used — nothing in the code enforces or infers this mapping.

`GestureActionDebouncer` sits between a per-frame gesture prediction and firing an action:
a gesture must be predicted for `stable_frames` consecutive frames (default 3) before it's
considered intentional, and a fired action won't repeat for the same gesture within
`cooldown_seconds` (default 1.0). Both are constructor args, not global state, specifically
so tests can use `cooldown_seconds=0` / small `stable_frames` for fast, deterministic
assertions instead of sleeping.

### `kalman.py`

A ~20-line hand-rolled linear Kalman filter (predict/update only), used purely to smooth
the drawn wrist position — it does not affect gesture classification. It replaces the
`filterpy` package, which is unmaintained and fails to build under modern setuptools; see
HISTORY.md for why. Same constructor/attribute shape as `filterpy.kalman.KalmanFilter`
(`dim_x`, `dim_z`, `.x .F .H .P .Q .R`, `.predict()`, `.update(z)`), so swapping back would
be a one-line import change if ever needed.

### Testing hardware/OS-dependent code

`cv2` (headless build), `numpy`, `pandas`, `torch`, and `torchvision` are real dependencies
in tests — tensor shapes, checkpoint round-trips, the training loop, and CSV parsing are
exercised for real, not mocked. `mediapipe` (needs a real trained detection model) and
`win32api`/`win32con` (Windows-only, cannot exist on non-Windows) are stubbed at import
time in `tests/conftest.py` only when the real package isn't importable, so pure
control-flow (e.g. "did the debounced action fire") can be tested without hardware or OS.
These stubs never assert anything about real hand-tracking accuracy or real OS key events —
that can only be verified on an actual Windows machine with a webcam.

### `win_control.py`

Thin wrapper over `win32api.keybd_event` for media keys and alt-tab. Windows-only;
`pywin32` is installed conditionally in `requirements.txt` via
`pywin32; platform_system == "Windows"`.
