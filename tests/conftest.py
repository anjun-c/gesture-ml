"""
Shared test setup.

This project depends on hardware/vendor SDKs that either can't run in CI
(mediapipe's real hand-tracking model, and win32api/win32con which are
Windows-only and cannot exist on this platform at all). Where the real
package isn't installed, we install a minimal stand-in into sys.modules
*before* the project modules are imported, so import-time `import mediapipe`
/ `import win32api` doesn't crash test collection. These stubs are only for
import safety and control-flow tests (e.g. "did the debounced action get
called") — they do not simulate real hand detection or real OS key events,
which can only be verified on an actual machine with a webcam (see tests'
module docstrings / README).
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _install_stub(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


try:
    import mediapipe  # noqa: F401
except ImportError:
    class _FakeHands:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, *_args, **_kwargs):
            return types.SimpleNamespace(multi_hand_landmarks=None)

    hands_mod = _install_stub("mediapipe.solutions.hands", Hands=_FakeHands, HAND_CONNECTIONS=[])
    drawing_mod = _install_stub("mediapipe.solutions.drawing_utils", draw_landmarks=lambda *a, **k: None)
    solutions_mod = _install_stub("mediapipe.solutions", hands=hands_mod, drawing_utils=drawing_mod)
    _install_stub("mediapipe", solutions=solutions_mod)

try:
    import win32api  # noqa: F401
except ImportError:
    _install_stub("win32api", keybd_event=lambda *a, **k: None)

try:
    import win32con  # noqa: F401
except ImportError:
    _install_stub(
        "win32con",
        VK_MEDIA_PLAY_PAUSE=0xB3,
        VK_MEDIA_NEXT_TRACK=0xB0,
        VK_MEDIA_PREV_TRACK=0xB1,
        VK_VOLUME_UP=0xAF,
        VK_VOLUME_DOWN=0xAE,
        VK_VOLUME_MUTE=0xAD,
        VK_MENU=0x12,
        VK_TAB=0x09,
        KEYEVENTF_EXTENDEDKEY=0x0001,
        KEYEVENTF_KEYUP=0x0002,
    )
