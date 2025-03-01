# Copyright (c) 2025, SountIO
import json
import os
import random

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from src.mfcc import extract_mfcc_features
from src.model import SnoreDetectionModel

CONFIG_PATH = "src/config.json"


class SnoreDataset(Dataset):
    def __init__(self, data_dir, config, augment=False):
        self.data_dir = data_dir
        self.config = config
        self.augment = augment
        self.files = []
        self.labels = []
        for label in ["snoring", "no_snoring"]:
            label_dir = os.path.join(data_dir, label)
            for file in os.listdir(label_dir):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(label_dir, file))
                    self.labels.append(1 if label == "snoring" else 0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        mfcc = extract_mfcc_features(audio_path, self.config)

        if self.augment:
            mfcc = self.apply_augmentation(mfcc, label)

        mfcc = self.normalize(mfcc)
        return mfcc, label

    def apply_augmentation(self, mfcc, label):
        """
        Apply augmentation by mixing snore and non-snore audio.

        Args:
            mfcc (torch.Tensor): MFCC features.
            label (int): Original label (1 for snore, 0 for no snore).

        Returns:
            torch.Tensor: Augmented MFCC features.
        """
        if label == 1:  # Snore
            non_snore_files = [f for f, l in zip(self.files, self.labels) if l == 0]
            if non_snore_files:
                non_snore_path = random.choice(non_snore_files)
                non_snore_mfcc = extract_mfcc_features(non_snore_path, self.config)
                mfcc = (mfcc + non_snore_mfcc) / 2
        else:  # No snore
            non_snore_files = [f for f, l in zip(self.files, self.labels) if l == 0]
            if non_snore_files:
                non_snore_path = random.choice(non_snore_files)
                non_snore_mfcc = extract_mfcc_features(non_snore_path, self.config)
                mfcc = (mfcc + non_snore_mfcc) / 2
        return mfcc

    def normalize(self, mfcc):
        """
        Normalize MFCC features.

        Args:
            mfcc (torch.Tensor): MFCC features.

        Returns:
            torch.Tensor: Normalized MFCC features.
        """
        mean = mfcc.mean()
        std = mfcc.std()
        return (mfcc - mean) / std


def train_model(model, dataloader, epochs=10, learning_rate=0.001):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for mfcc_batch, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(mfcc_batch)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = SnoreDetectionModel(CONFIG_PATH)
    dataset = SnoreDataset("data", config, augment=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train_model(model, dataloader)


if __name__ == "__main__":
    main()
