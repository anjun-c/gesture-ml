# HISTORY.md

A changelog of substantive changes, architectural decisions, and the reasoning behind
them, kept for future contributors (human or Claude). Ordinary commit messages cover the
mechanics; this file covers *why*, especially for decisions that were explicitly checked
with the project owner rather than made unilaterally.

## 2026-08-23 — Repository review: bugfixes, train/inference split, gesture wiring

Starting state: `src/` held four hand-written `.py` scripts each paired with a
hand-duplicated `.ipynb` notebook (`capture`, `detect`, `gesture_rec`, `win_control`), no
tests, no dependency manifest, and no `__main__` guards.

### Bugs found and fixed

- **`win_control.py` — `NameError` on `win32con`.** `alt_tab_quick()`,
  `alt_tab_scroll_press()`, and `alt_tab_scroll_release()` referenced `win32con.VK_MENU`
  etc., but only specific names had been imported (`from win32con import (...)`) —
  `win32con` itself was never imported. Fixed by adding `import win32con`.
- **Train/inference landmark feature-order mismatch.** Training-data extraction
  (`extract_landmarks_from_image`) built each 42-value landmark vector as all 21
  x-coordinates followed by all 21 y-coordinates. The live-inference path
  (`gesture_recognition_integration` in the old `detect.ipynb`) instead flattened
  `[[x,y], [x,y], ...]`, i.e. interleaved `x0,y0,x1,y1,...`. Same length, wrong order —
  this would not crash, it would just silently feed the model scrambled input at
  inference time and degrade predictions. Fixed by making inference build the vector the
  same way training does; a regression test
  (`test_gesture_recognition_integration_uses_x_then_y_feature_order`) now pins this down.
- **`gesture_rec.ipynb` — `GestureLandmarkDataset(DataLoader)`.** The notebook version
  subclassed `torch.utils.data.DataLoader` instead of `Dataset` (the `.py` version had it
  right), and was later wrapped in a real `DataLoader(...)` — a real bug introduced by the
  notebook/`.py` drift. Resolved by dropping notebooks (see below) and keeping the
  `Dataset` version.
- **`gesture_rec.py` — model/data shape mismatch.** `train_gesture_model()` called
  `load_gesture_data()`, which loaded raw 50×50×3 *images*, while
  `model.fc1 = nn.Linear(42, 128)` expected flattened *landmark* vectors — an immediate
  shape-mismatch crash if run. The dataset-loading and training code were two competing,
  inconsistent pipelines living in the same file. Resolved by the train/inference split
  below, which keeps exactly one training pipeline (landmarks → CSV → `GestureLandmarkDataset`).
- **`num_classes=20` hardcoded** in the model constructor, disconnected from whatever
  dataset was actually loaded. Now derived from `len(classes)` at training time and
  carried inside the saved checkpoint.

### Architectural decisions (checked with the project owner via AskUserQuestion)

1. **Train/inference split → chose "split into `train.py` + `model.py`".**
   Previously, `detect.ipynb` did `from gesture_rec import gesture_recognition_integration`,
   and `gesture_rec.py`/`.ipynb` had no `__main__` guard and no checkpointing — so merely
   *importing* the inference helper re-ran the entire dataset scan and training loop, and
   required the dataset directory to exist just to do live detection. Fixed by splitting
   into `model.py` (import-safe: model class + inference helpers only) and `train.py`
   (data prep + training, `if __name__ == "__main__"`-guarded, saves a
   `{"model_state_dict", "classes"}` checkpoint via `torch.save`). `detect.py` now loads a
   pretrained checkpoint once at startup via `model.load_model()` instead of retraining.
2. **Notebook/`.py` sync → chose "keep `.py` as source of truth, drop notebooks".**
   The hand-duplicated `.ipynb` files had already drifted from their `.py` counterparts
   (that's how the `GestureLandmarkDataset(DataLoader)` bug happened) — maintaining two
   copies by hand was actively causing bugs. All four `.ipynb` files were deleted; the
   `.py` files (still using `# %%` cell markers, openable as notebooks via Jupytext/VS
   Code) are now the only copy.
3. **Gesture → action wiring → chose "yes, wire a basic gesture-to-action map".**
   Previously `predicted_gesture` was computed and only printed — no gesture ever actually
   did anything. Added `GESTURE_ACTIONS` (a `dict[gesture_name, win_control_function]`) and
   `GestureActionDebouncer` in `detect.py`, which requires a gesture to be predicted for
   several consecutive frames before acting (avoids single-frame jitter) and rate-limits
   repeated firing of the same action (avoids e.g. spamming volume-up while a gesture is
   held). `GESTURE_ACTIONS` keys are placeholders (`"palm"`, `"fist"`, etc.) — they must be
   edited to match whatever class/folder names the user's actual dataset used; nothing
   infers this mapping automatically.
4. **Dependency manifest → chose "yes, plain `requirements.txt`".** Added, with
   `pywin32` conditional on `platform_system == "Windows"` since the rest of the pipeline
   (training, landmark extraction) is platform-independent.

## 2026-08-23 — Test suite

Added a `pytest` suite (`tests/`, 24 tests at this point) covering the logic that doesn't
require a camera, a trained model, or Windows: model forward-pass shape, checkpoint
save/load round-trip, the train/inference feature-order fix above, `EarlyStopping`, the
landmark CSV dataset, the training loop end-to-end on real (synthetic) tensors, webcam
init/capture/release failure paths, and the gesture debounce/action-wiring logic.

Real `numpy`/`pandas`/`torch`/`torchvision`/`opencv-python` (headless build) are used in
tests so the actual math runs, not mocks. `mediapipe` and `win32api`/`win32con` are stubbed
in `tests/conftest.py` only when the real package can't be imported — `win32api` literally
cannot exist on a non-Windows test runner, and asserting on MediaPipe's real hand-tracking
output requires real camera images this suite doesn't have. These stubs verify wiring
("did the mapped action get called"), not real hand-tracking accuracy or real OS key events.

**Bug the test suite caught:** `GestureLandmarkDataset.__getitem__` used `row[-1]` /
`row[:-1]` — positional indexing on a pandas `Series` whose index is the string column
names (`lm_0`, …, `label`). Current pandas raises `KeyError` on this rather than falling
back to positional lookup (an old, since-removed fallback). Fixed to `row.iloc[-1]` /
`row.iloc[:-1]`. This bug predated the test suite (it was already present in the original
`gesture_rec.py`/`.ipynb`) and would have broken any training run on a recent pandas
version.

## 2026-08-23 — Replaced `filterpy` with a small built-in Kalman filter

While installing dependencies to run the test suite, `pip install filterpy` failed:
`filterpy` is unmaintained (last released 2018) and its `setup.py` uses the
now-removed `install_layout` `distutils` option, which breaks under current `setuptools`
(confirmed failing here with `setuptools==84.0.0`). This is a real installability risk for
anyone setting the project up with a reasonably current Python toolchain, not just this
sandbox.

**Architectural decision → chose "replace `filterpy` with a small built-in filter".**
`detect.py` only used `filterpy.kalman.KalmanFilter` for a fixed-matrix constant-velocity
predict/update cycle smoothing the drawn wrist position (visual only — it does not affect
gesture classification). Added `src/kalman.py`, a ~20-line numpy-only implementation with
the same constructor/attribute shape (`dim_x`, `dim_z`, `.x .F .H .P .Q .R`, `.predict()`,
`.update(z)`), so swapping back would be a one-line import change if ever needed. Covered
by `tests/test_kalman.py`. `filterpy` removed from `requirements.txt`; its stub in
`tests/conftest.py` removed since it's no longer needed.

## Known gaps (not yet done)

- No trained model ships with the repo — `gesture_model.pt` must be produced by running
  `train.py` against a real gesture-image dataset the user supplies (not included; see
  README for the expected `ImageFolder` layout).
- `GESTURE_ACTIONS` in `detect.py` still uses placeholder gesture names and must be edited
  to match the class names of whatever dataset is actually used for training.
- End-to-end behavior (a real hand gesture, seen by a real webcam, changing something on a
  real Windows machine) has not been verified — it can't be, from a Linux sandbox with no
  camera and no Windows. See tests' module docstrings for exactly what is and isn't covered.
