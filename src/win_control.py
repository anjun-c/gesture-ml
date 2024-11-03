# %%
import win32api
from win32con import (
    VK_MEDIA_PLAY_PAUSE, VK_MEDIA_NEXT_TRACK, VK_MEDIA_PREV_TRACK,
    VK_VOLUME_UP, VK_VOLUME_DOWN, VK_VOLUME_MUTE, KEYEVENTF_EXTENDEDKEY
)

# %%
def play_pause_media():
    win32api.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY, 0)
    win32api.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY | 0x0002, 0)

def next_track():
    win32api.keybd_event(VK_MEDIA_NEXT_TRACK, 0, KEYEVENTF_EXTENDEDKEY, 0)
    win32api.keybd_event(VK_MEDIA_NEXT_TRACK, 0, KEYEVENTF_EXTENDEDKEY | 0x0002, 0)

def previous_track():
    win32api.keybd_event(VK_MEDIA_PREV_TRACK, 0, KEYEVENTF_EXTENDEDKEY, 0)
    win32api.keybd_event(VK_MEDIA_PREV_TRACK, 0, KEYEVENTF_EXTENDEDKEY | 0x0002, 0)

def volume_up():
    win32api.keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_EXTENDEDKEY, 0)
    win32api.keybd_event(VK_VOLUME_UP, 0, KEYEVENTF_EXTENDEDKEY | 0x0002, 0)

def volume_down():
    win32api.keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_EXTENDEDKEY, 0)
    win32api.keybd_event(VK_VOLUME_DOWN, 0, KEYEVENTF_EXTENDEDKEY | 0x0002, 0)

def mute_volume():
    win32api.keybd_event(VK_VOLUME_MUTE, 0, KEYEVENTF_EXTENDEDKEY, 0)
    win32api.keybd_event(VK_VOLUME_MUTE, 0, KEYEVENTF_EXTENDEDKEY | 0x0002, 0)

def alt_tab_quick():
    # Press Alt key
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
    # Press Tab key
    win32api.keybd_event(win32con.VK_TAB, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
    # Release Tab key
    win32api.keybd_event(win32con.VK_TAB, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0)
    # Release Alt key
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0)

def alt_tab_scroll_press():
    # Press Alt key
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)
    # Press Tab key
    win32api.keybd_event(win32con.VK_TAB, 0, win32con.KEYEVENTF_EXTENDEDKEY, 0)

def alt_tab_scroll_release():
    # Release Tab key
    win32api.keybd_event(win32con.VK_TAB, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0)
    # Release Alt key
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_EXTENDEDKEY | win32con.KEYEVENTF_KEYUP, 0)


