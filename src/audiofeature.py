# Copyright (c) 2025, SountIO
import json

import librosa
import numpy as np
import torch

MS_PER_SEC = 1000


class FeatureConfig:
    def __init__(self, config):
        self.sample_rate = config["feature_extraction"]["sample_rate"]
        self.audio_length_sec = config["feature_extraction"]["audio_length_sec"]
        self.n_mfcc = config["feature_extraction"].get("n_mfcc", 40)
        self.n_mels = config["feature_extraction"].get("n_mels", 128)
        self.window_size_ms = config["feature_extraction"]["window_size_ms"]
        self.window_step_ms = config["feature_extraction"]["window_step_ms"]
        self.method = config["feature_extraction"].get("method", "mfcc")

    @classmethod
    def from_json(cls, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config)


def get_audio_feature(audio_path, config_path):
    """
    Get audio features based on the specified feature extraction method in the config.
    Returns:
        torch.Tensor: Extracted audio features.
    """
    config = FeatureConfig.from_json(config_path)
    audio_length_sec = config.audio_length_sec
    sample_rate = config.sample_rate
    audio, _ = librosa.load(audio_path, sr=sample_rate, duration=audio_length_sec)

    if config.method == "mfcc":
        return _extract_mfcc_features(audio, config)

    if config.method == "mel_spectrogram":
        return _extract_mel_spectrogram(audio, config)

    raise ValueError(f"Unsupported feature extraction method: {config.method}")


def _extract_mfcc_features(audio, config):
    sample_rate = config.sample_rate
    n_mfcc = config.n_mfcc
    window_size_ms = config.window_size_ms
    window_step_ms = config.window_step_ms

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=int(sample_rate * window_size_ms / MS_PER_SEC),
        hop_length=int(sample_rate * window_step_ms / MS_PER_SEC),
    )

    mfcc_db = librosa.amplitude_to_db(mfcc, ref=np.max)
    mfcc_tensor = torch.tensor(mfcc_db, dtype=torch.float32).unsqueeze(0)
    return _normalize_db(mfcc_tensor)


def _extract_mel_spectrogram(audio, config):
    sample_rate = config.sample_rate
    n_mels = config.n_mels
    window_size_ms = config.window_size_ms
    window_step_ms = config.window_step_ms

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
        mel_spectrogram_tensor = _normalize_db(mel_spectrogram_tensor)

    return mel_spectrogram_tensor


def _normalize_db(tensor):
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


def _plot_audio_features(audio_paths, config_path, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))
    config = FeatureConfig.from_json(config_path)
    for i, audio_path in enumerate(audio_paths):
        try:
            features_tensor = get_audio_feature(audio_path, config_path)
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


def _example_usage():
    import os

    from model import CONFIG_PATH

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

    _plot_audio_features(no_snore_files, CONFIG_PATH, "No Snore")
    _plot_audio_features(snore_files, CONFIG_PATH, "Snore")


if __name__ == "__main__":
    _example_usage()
