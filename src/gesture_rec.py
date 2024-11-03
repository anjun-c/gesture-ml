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
# from https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset

def load_gesture_data(data_dir, batch_size=900, image_size=(50, 50)):
    """
    Load image data from a directory, applying necessary transforms.
    Args:
    - data_dir: the path to the data directory.
    - batch_size: number of samples per batch.
    - image_size: size to resize the images.

    Returns:
    - DataLoader for training and validation sets.
    """
    
    # Define the transformations for the training and validation data
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),  # Augmentation for training data
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with mean and std dev
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load datasets using ImageFolder (it automatically assigns labels based on folder names)
    image_datasets = {
        'train': datasets.ImageFolder(root=f"../data/archive/train", transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=f"../data/archive/test", transform=data_transforms['val'])
    }

    # Create data loaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False)
    }

    return dataloaders

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


