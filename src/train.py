# Copyright (c) 2025, SountIO
import gc
import json
import os
from datetime import datetime
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.audiofeature import get_audio_feature
from src.model import CONFIG_PATH, SnoreDetectionModel

CKPT_DIR = "ckpt"
TRAIN_DIR = os.path.join("data", "train")
VAL_DIR = os.path.join("data", "val")


class SnoreDataset(Dataset):
    def __init__(self, data_dir: str, config_path: str):
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

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        audio_path = self.files[idx]
        label = self.labels[idx]
        audio_feature = get_audio_feature(audio_path, self.config_path)

        return audio_feature, label


class TrainingDetails:
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.val_loss = val_loss
        self.val_accuracy = val_accuracy

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
        }


def _train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
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


def _evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
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


def _save_model(
    model: torch.nn.Module,
    epochs: int,
    config_path: str,
    training_details: TrainingDetails,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subfolder = os.path.join(CKPT_DIR, f"ckpt_{timestamp}")
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    # Save model checkpoint
    ckpt_path = os.path.join(subfolder, f"snore_detection_model_epochs{epochs}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to folder: {subfolder}")

    # Copy config file
    config_dest_path = os.path.join(subfolder, "config.json")
    _copy_config_file(config_path, config_dest_path)

    # Save model details
    details_path = os.path.join(subfolder, "training_details.json")
    _save_training_details(details_path, training_details)


def _copy_config_file(src_path: str, dest_path: str) -> None:
    with open(src_path, "r", encoding="utf-8") as src_file:
        config_data = src_file.read()
    with open(dest_path, "w", encoding="utf-8") as dest_file:
        dest_file.write(config_data)


def _save_training_details(
    details_path: str, training_details: TrainingDetails
) -> None:
    with open(details_path, "w", encoding="utf-8") as details_file:
        json.dump(training_details.to_dict(), details_file, indent=4)


def _train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
) -> None:
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

    training_details = TrainingDetails(
        batch_size=train_loader.batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
    )

    _save_model(model, epochs, CONFIG_PATH, training_details)


def _cleanup_resources() -> None:
    gc.collect()

    # If CUDA is available, clear the GPU memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all CUDA operations are completed
        print("CUDA resources cleared.")
    else:
        print("No CUDA device available, skipping CUDA cleanup.")


def main() -> None:
    batch_size = 8
    num_epochs = 2
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
