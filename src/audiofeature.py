# Copyright (c) 2025, SountIO

from typing import List

import librosa
import numpy as np
import torch

from src.config import FeatureConfig

MS_PER_SEC = 1000


class AudioFeatureExtractor:
    def __init__(self):
        self.config = FeatureConfig.get_config()

    def load_config(self):
        self.config = FeatureConfig.get_config()

    def get_feature_from_file(self, audio_path: str) -> torch.Tensor:
        """
        Get audio features based on the specified feature extraction method in the config.
        Returns:
            torch.Tensor: Extracted audio features.
        """
        audio_length_sec = self.config.audio_length_sec
        sample_rate = self.config.sample_rate
        audio, _ = librosa.load(audio_path, sr=sample_rate, duration=audio_length_sec)

        if self.config.method == "mfcc":
            return self._extract_mfcc_features(audio)

        if self.config.method == "mel_spectrogram":
            return self._extract_mel_spectrogram(audio)

        raise ValueError(f"Unsupported feature extraction method: {self.config.method}")

    def get_feature_from_stream(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> torch.Tensor:
        """
        Get audio features from streaming audio data based on the specified feature extraction method in the config.
        Args:
            audio_data (np.ndarray): Audio data as a numpy array.
            sample_rate (int): Sample rate of the audio data.
        Returns:
            torch.Tensor: Extracted audio features.
        """
        if sample_rate != self.config.sample_rate:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=self.config.sample_rate
            )
            sample_rate = self.config.sample_rate

        audio_length_sec = self.config.audio_length_sec

        # Ensure the audio data is the correct length
        if len(audio_data) > sample_rate * audio_length_sec:
            audio_data = audio_data[: sample_rate * audio_length_sec]
        elif len(audio_data) < sample_rate * audio_length_sec:
            padding = sample_rate * audio_length_sec - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), "constant")

        if self.config.method == "mfcc":
            return self._extract_mfcc_features(audio_data)

        if self.config.method == "mel_spectrogram":
            return self._extract_mel_spectrogram(audio_data)

        raise ValueError(f"Unsupported feature extraction method: {self.config.method}")

    def _extract_mfcc_features(self, audio: np.ndarray) -> torch.Tensor:
        sample_rate = self.config.sample_rate
        n_mfcc = self.config.n_mfcc
        window_size_ms = self.config.window_size_ms
        window_step_ms = self.config.window_step_ms

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=int(sample_rate * window_size_ms / MS_PER_SEC),
            hop_length=int(sample_rate * window_step_ms / MS_PER_SEC),
        )

        mfcc_db = librosa.amplitude_to_db(mfcc, ref=np.max)
        mfcc_tensor = torch.tensor(mfcc_db, dtype=torch.float32).unsqueeze(0)
        return self._normalize_db(mfcc_tensor)

    def _extract_mel_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        sample_rate = self.config.sample_rate
        n_mels = self.config.n_mels
        window_size_ms = self.config.window_size_ms
        window_step_ms = self.config.window_step_ms

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=int(sample_rate * window_size_ms / MS_PER_SEC),
            hop_length=int(sample_rate * window_step_ms / MS_PER_SEC),
        )

        if np.all(mel_spectrogram == 0):
            mel_spectrogram_tensor = torch.zeros(
                (1, n_mels, mel_spectrogram.shape[1]), dtype=torch.float32
            )
        else:
            mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
            mel_spectrogram_tensor = torch.tensor(
                mel_spectrogram_db, dtype=torch.float32
            ).unsqueeze(0)
            mel_spectrogram_tensor = self._normalize_db(mel_spectrogram_tensor)

        return mel_spectrogram_tensor

    def _normalize_db(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize the tensor to have zero mean.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / std


def _plot_audio_features(audio_paths: List[str], title: str) -> None:
    import matplotlib.pyplot as plt

    feature_extractor = AudioFeatureExtractor()

    plt.figure(figsize=(15, 10))
    config = FeatureConfig.get_config()
    for i, audio_path in enumerate(audio_paths):
        try:
            features_tensor = feature_extractor.get_feature_from_file(audio_path)
            features_np = features_tensor.squeeze(0).numpy()

            window_step_ms = config.window_step_ms
            num_frames = features_np.shape[1]
            time_axis = np.arange(num_frames) * (window_step_ms / MS_PER_SEC)

            plt.subplot(2, 4, i + 1)
            plt.imshow(
                features_np,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                extent=[time_axis[0], time_axis[-1], 0, features_np.shape[0]],
            )
            plt.colorbar(label="Magnitude (dB)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Coefficients")
            plt.title(f"{title} {i + 1}")
        except FileNotFoundError:
            print(
                f"Audio file not found at {audio_path}. Please ensure the file exists."
            )
        except Exception as e:
            print(f"An error occurred while processing {audio_path}: {e}")

    plt.tight_layout()
    plt.show()


def _example_usage() -> None:
    import os

    train_dir = os.path.join("data", "train")

    no_snore_files = [
        os.path.join(train_dir, "no_snore", f)
        for f in os.listdir(os.path.join(train_dir, "no_snore"))[:4]
    ]
    snore_files = [
        os.path.join(train_dir, "snore", f)
        for f in os.listdir(os.path.join(train_dir, "snore"))[:4]
    ]

    print(no_snore_files)
    print(snore_files)

    _plot_audio_features(no_snore_files, "No Snore")
    _plot_audio_features(snore_files, "Snore")


if __name__ == "__main__":
    _example_usage()
