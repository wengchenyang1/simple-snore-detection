# Copyright (c) 2025, SountIO
import json
import os

import librosa
import numpy as np
import torch

MS_PER_SEC = 1000


def extract_mfcc_features(audio_path, config):
    """
    Extract MFCC features from an audio file based on config parameters.

    Args:
        audio_path (str): Path to the audio file.
        config (dict): Configuration dictionary with feature_extraction parameters.

    Returns:
        torch.Tensor: MFCC features with shape (1, n_mfcc, num_frames).
    """
    # Extract feature extraction parameters
    sample_rate = config["feature_extraction"]["sample_rate"]
    audio_length_sec = config["feature_extraction"]["audio_length_sec"]
    n_mfcc = config["feature_extraction"]["n_mfcc"]
    window_size_ms = config["feature_extraction"]["window_size_ms"]
    window_step_ms = config["feature_extraction"]["window_step_ms"]

    # Load the audio file
    audio, _ = librosa.load(audio_path, sr=sample_rate, duration=audio_length_sec)

    # Compute MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=int(sample_rate * window_size_ms / MS_PER_SEC),
        hop_length=int(sample_rate * window_step_ms / MS_PER_SEC),
    )

    # Convert MFCC to decibel scale
    mfcc_db = librosa.amplitude_to_db(mfcc, ref=np.max)

    # Convert to PyTorch tensor and add channel dimension, shape: (1, n_mfcc, num_frames)
    mfcc_tensor = torch.tensor(mfcc_db, dtype=torch.float32).unsqueeze(0)
    return mfcc_tensor


def _plot_mfcc_features(audio_paths, config, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    for i, audio_path in enumerate(audio_paths):
        try:
            # Extract MFCC features
            mfcc_tensor = extract_mfcc_features(audio_path, config)
            mfcc_np = mfcc_tensor.squeeze(0).numpy()  # Shape: (n_mfcc, num_frames)

            # Calculate time axis for plotting
            window_step_ms = config["feature_extraction"]["window_step_ms"]
            num_frames = mfcc_np.shape[1]
            time_axis = np.arange(num_frames) * (window_step_ms / MS_PER_SEC)

            # Plot the MFCC features
            plt.subplot(2, 4, i + 1)
            plt.imshow(
                mfcc_np,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                extent=[time_axis[0], time_axis[-1], 0, mfcc_np.shape[0]],
            )
            plt.colorbar(label="MFCC Magnitude (dB)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("MFCC Coefficients")
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
    CONFIG_PATH = "src/config.json"
    TRAIN_DIR = os.path.join("data", "train")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Get the first 4 files from "no_snore" and "snore" categories
    no_snore_files = [
        os.path.join(TRAIN_DIR, "no_snore", f)
        for f in os.listdir(os.path.join(TRAIN_DIR, "no_snore"))[:4]
    ]
    snore_files = [
        os.path.join(TRAIN_DIR, "snore", f)
        for f in os.listdir(os.path.join(TRAIN_DIR, "snore"))[:4]
    ]

    # Plot MFCC features for "no_snore" and "snore" files
    _plot_mfcc_features(no_snore_files, config, "No Snore")
    _plot_mfcc_features(snore_files, config, "Snore")

if __name__ == "__main__":
    _example_usage()
