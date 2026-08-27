"""
capture.py tests. This sandbox genuinely has no webcam, so
test_initialize_webcam_no_camera exercises the real failure path with the
real cv2 (not a mock) -- it's the one case here we can honestly verify
end-to-end without hardware.
"""
from unittest.mock import Mock

import cv2

from capture import capture_video, initialize_webcam, release_resources


def test_initialize_webcam_no_camera_returns_none():
    # No real camera device exists in this environment, so this hits cv2's
    # actual "failed to open" path rather than a mock.
    assert initialize_webcam() is None


def test_capture_video_returns_none_on_failed_read():
    fake_cap = Mock()
    fake_cap.read.return_value = (False, None)
    assert capture_video(fake_cap) is None


def test_capture_video_returns_frame_on_success():
    fake_cap = Mock()
    fake_cap.read.return_value = (True, "frame-data")
    assert capture_video(fake_cap) == "frame-data"


def test_release_resources_releases_capture(monkeypatch):
    # cv2.destroyAllWindows() needs a GUI backend (Windows/GTK/Cocoa); this
    # sandbox only has headless opencv installed for testing, so the actual
    # window-toolkit call is stubbed here. The real project depends on
    # non-headless opencv-python (see requirements.txt), where this runs for real.
    monkeypatch.setattr(cv2, "destroyAllWindows", Mock())
    fake_cap = Mock()
    release_resources(fake_cap)
    fake_cap.release.assert_called_once()
