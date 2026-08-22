"""Gesture recognition model definition and inference helpers.

This module is safe to import: it has no top-level dataset loading or
training side effects. For training, see train.py.
"""
import cv2
import torch
import torch.nn as nn

DEFAULT_MODEL_PATH = "gesture_model.pt"


def extract_landmarks_from_image(image, hands_instance):
    """
    Extract landmarks from the given image using MediaPipe.
    Args:
        image: The input image.
        hands_instance: An instance of MediaPipe Hands solution.

    Returns:
        A flattened list of landmarks (x, y coordinates) or None if no hands detected.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_instance.process(rgb_image)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
        return landmarks
    return None


class GestureRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(GestureRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(42, 128)  # 21 landmarks * 2 (x and y)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def load_model(checkpoint_path=DEFAULT_MODEL_PATH, device="cpu"):
    """
    Load a trained GestureRecognitionModel checkpoint produced by train.py.

    Returns:
        (model, classes): the loaded model in eval mode, and the list of
        class names (index-aligned with the model's output logits).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    model = GestureRecognitionModel(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def recognize_gesture(model, landmarks):
    """Run inference on a single 42-value landmark vector, returning the predicted class index."""
    model.eval()
    with torch.no_grad():
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        output = model(landmarks_tensor)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()


def gesture_recognition_integration(hand_landmarks, model, classes):
    """
    Given MediaPipe hand landmarks, predict the gesture and return its class name (or None).
    """
    if hand_landmarks is None:
        return None
    landmarks_array = (
        [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
    )
    predicted_idx = recognize_gesture(model, landmarks_array)
    return classes[predicted_idx]
