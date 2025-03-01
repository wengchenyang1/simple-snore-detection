# Copyright (c) 2025, SountIO
import gc
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset

from src.audiofeature import get_audio_feature
from src.model import SnoreDetectionModel

CONFIG_PATH = "src/config.json"
CKPT_DIR = "ckpt"
TRAIN_DIR = os.path.join("data", "train")
VAL_DIR = os.path.join("data", "val")


class SnoreDataset(Dataset):
    def __init__(self, data_dir, config_path):
        self.data_dir = data_dir
        self.config_path = config_path
        self.files = []
        self.labels = []
        for label in ["snore", "no_snore"]:
            label_dir = os.path.join(data_dir, label)
            for file in os.listdir(label_dir):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(label_dir, file))
                    self.labels.append(1 if label == "snore" else 0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        audio_feature = get_audio_feature(audio_path, self.config_path)

        return audio_feature, label


def _train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}!")
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epochs):
        train_loss, train_accuracy = _train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_accuracy = _evaluate_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    _save_model(model, epochs)


def _train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for audio_feature_batch, labels in dataloader:
        audio_feature_batch, labels = audio_feature_batch.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(audio_feature_batch)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predictions = (outputs.squeeze() > 0.5).float()
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy


def _evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for audio_feature_batch, labels in dataloader:
            audio_feature_batch, labels = audio_feature_batch.to(device), labels.to(
                device
            )
            outputs = model(audio_feature_batch)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item()

            predictions = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy


def _save_model(model, epochs):
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(
        CKPT_DIR, f"snore_detection_model_{timestamp}_epochs{epochs}.pth"
    )
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


def _cleanup_resources():
    gc.collect()

    # If CUDA is available, clear the GPU memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all CUDA operations are completed
        print("CUDA resources cleared.")
    else:
        print("No CUDA device available, skipping CUDA cleanup.")


def main():
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    num_workers = 4

    _cleanup_resources()

    model = SnoreDetectionModel(CONFIG_PATH)
    train_dataset = SnoreDataset(TRAIN_DIR, CONFIG_PATH)
    val_dataset = SnoreDataset(VAL_DIR, CONFIG_PATH)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    _train_model(
        model, train_loader, val_loader, epochs=num_epochs, learning_rate=learning_rate
    )

    _cleanup_resources()


if __name__ == "__main__":
    main()
