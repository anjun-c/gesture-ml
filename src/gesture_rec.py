# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# %%
# Define the CNN model
class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(21 * 2, 128)  # 21 hand landmarks (x, y) pairs
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Assuming 3 classes: open hand, fist, swipe

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
model = GestureRecognitionModel()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# %%
# dummy
def load_gesture_data():
    data = np.random.randn(100, 21, 2)  
    labels = np.random.randint(0, 3, 100)  
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# %%
# Training loop
def train_gesture_model(epochs=10):
    model.train()
    for epoch in range(epochs):
        # Load gesture data
        data, labels = load_gesture_data()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# %%
# Gesture recognition function (for real-time prediction)
def recognize_gesture(landmarks):
    model.eval()
    with torch.no_grad():
        landmarks = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        output = model(landmarks)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

# %%
# Example use in the main detection loop
def gesture_recognition_integration(hand_landmarks):
    if hand_landmarks:
        landmarks_array = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
        predicted_gesture = recognize_gesture(landmarks_array)
        return predicted_gesture
    return None


