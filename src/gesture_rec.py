# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np

# %%
# Define the CNN model
num_classes = 20

class GestureRecognitionModel(nn.Module):
    def __init__(self):
        super(GestureRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(7500, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
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
            transforms.RandomResizedCrop(50),
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
        'train': datasets.ImageFolder(root=data_dir + "/train", transform=data_transforms['train']),
        'val': datasets.ImageFolder(root=data_dir + "/test", transform=data_transforms['val'])
    }

    # Create data loaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False)
    }

    return dataloaders

# %%
load_gesture_data("../data/archive", 900, (50,50))

# %%
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False

# %%
# Training loop
def train_gesture_model(epochs=10, patience=4):
    model.train()
    dataloaders = load_gesture_data("../data/archive", 900, (50,50))
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for data, labels in train_loader:
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Check early stopping
        if early_stopping(val_loss):
            print("Early stopping")
            break

# %%
train_gesture_model()

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


