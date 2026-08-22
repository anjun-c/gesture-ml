# gesture-ml
for fun

Webcam hand-gesture recognition (MediaPipe + a small PyTorch classifier) wired up to
Windows media-key controls.

## Setup

```
pip install -r requirements.txt
```

## Usage

1. Train a model on a gesture image dataset laid out as `<data-dir>/train/<class>/*.jpg`
   and `<data-dir>/test/<class>/*.jpg` (e.g. the Kaggle "hand-gesture-recognition-dataset"):

   ```
   cd src
   python train.py --data-dir ../data/archive
   ```

   This saves a checkpoint to `gesture_model.pt`.

2. Edit `GESTURE_ACTIONS` in `src/detect.py` so its keys match the class/folder names
   your dataset actually used, mapped to the `win_control` action you want each gesture
   to trigger.

3. Run live detection (Windows only, for the media-key controls):

   ```
   python detect.py
   ```
