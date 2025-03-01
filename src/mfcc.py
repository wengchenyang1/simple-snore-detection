# Copyright (c) 2025, SountIO
import librosa
import torch


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
    audio_length = config["feature_extraction"]["audio_length"]
    n_mfcc = config["feature_extraction"]["n_mfcc"]
    window_size_ms = config["feature_extraction"]["window_size_ms"]
    window_step_ms = config["feature_extraction"]["window_step_ms"]

    # Load the audio file
    audio, _ = librosa.load(audio_path, sr=sample_rate, duration=audio_length)

    # Compute MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=int(sample_rate * window_size_ms / 1000),
        hop_length=int(sample_rate * window_step_ms / 1000),
    )

    # Convert to PyTorch tensor and add channel dimension
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(
        0
    )  # Shape: (1, n_mfcc, num_frames)
    return mfcc_tensor
