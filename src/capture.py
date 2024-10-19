# %%
import cv2

# %%
def initialize_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("err")
        return None
    return cap

# %%
def display_frame(frame, window_name='detect'):
    cv2.imshow(window_name, frame)

# %%
def capture_video(cap):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None
    return frame

# %%
def release_resources(cap):
    cap.release()
    cv2.destroyAllWindows()


