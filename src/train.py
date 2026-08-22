"""
Train the gesture recognition model on a landmark dataset derived from images.

Usage:
    python train.py [--data-dir ../data/archive] [--model-path gesture_model.pt] [--epochs 10]

Expects a directory structure of:
    <data-dir>/train/<class_name>/*.jpg
    <data-dir>/test/<class_name>/*.jpg
(the ImageFolder convention), e.g. the Kaggle "hand-gesture-recognition-dataset".
"""
import argparse
import csv

import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from model import GestureRecognitionModel, extract_landmarks_from_image


class GestureLandmarkDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        landmarks = row[:-1].values.astype(np.float32)
        label = int(row[-1])
        return torch.tensor(landmarks), torch.tensor(label)


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

        return self.counter >= self.patience


def extract_landmarks_to_csv(image_dataset, hands_instance, output_csv_path):
    """Run MediaPipe over every image in an ImageFolder dataset and write landmarks + label to CSV."""
    with open(output_csv_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        header = [f"lm_{i}" for i in range(42)] + ["label"]
        csv_writer.writerow(header)

        for i, (img, label) in enumerate(image_dataset):
            img = transforms.ToPILImage()(img)
            img = np.array(img)

            landmarks = extract_landmarks_from_image(img, hands_instance)
            if landmarks is not None:
                csv_writer.writerow(landmarks + [label])

            if i % 100 == 0:
                print(f"Processed {i}/{len(image_dataset)} images from {output_csv_path}")

    print(f"Finished extracting landmarks to {output_csv_path}")


def train_gesture_model(model, train_loader, val_loader, epochs=10, patience=4, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        if early_stopping(val_loss):
            print("Early stopping")
            break

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../data/archive")
    parser.add_argument("--model-path", default="gesture_model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    hands_instance = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    train_image_dataset = datasets.ImageFolder(root=f"{args.data_dir}/train", transform=transforms.ToTensor())
    val_image_dataset = datasets.ImageFolder(root=f"{args.data_dir}/test", transform=transforms.ToTensor())

    # ImageFolder assigns class indices alphabetically; save the mapping so
    # inference (detect.py) can translate predicted indices back to gesture names.
    classes = train_image_dataset.classes

    extract_landmarks_to_csv(train_image_dataset, hands_instance, "landmarks_dataset.csv")
    extract_landmarks_to_csv(val_image_dataset, hands_instance, "landmarks_val_dataset.csv")

    train_loader = DataLoader(GestureLandmarkDataset("landmarks_dataset.csv"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(GestureLandmarkDataset("landmarks_val_dataset.csv"), batch_size=args.batch_size, shuffle=False)

    model = GestureRecognitionModel(num_classes=len(classes))
    train_gesture_model(model, train_loader, val_loader, epochs=args.epochs, patience=args.patience)

    torch.save({"model_state_dict": model.state_dict(), "classes": classes}, args.model_path)
    print(f"Saved trained model to {args.model_path}")


if __name__ == "__main__":
    main()
