# Copyright (c) 2025, SountIO
import argparse
import os

import librosa
import numpy as np
import torch
from torch import nn

from src.audiofeature import AudioFeatureExtractor
from src.config import FeatureConfig, ModelConfig
from src.model import SnoreDetectionModel


def load_model(
    ckpt_path: str, model_config: ModelConfig, feature_config: FeatureConfig
) -> nn.Module:
    model = SnoreDetectionModel(model_config, feature_config)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def infer(
    audio_data: np.ndarray,
    sample_rate: int,
    model: nn.Module,
    feature_extractor: AudioFeatureExtractor,
) -> float:
    # Resample if needed.
    if sample_rate != feature_extractor.config.sample_rate:
        audio_data = librosa.resample(
            audio_data,
            orig_sr=sample_rate,
            target_sr=feature_extractor.config.sample_rate,
        )
        sample_rate = feature_extractor.config.sample_rate

    audio_feature = feature_extractor.get_feature_from_stream(
        audio_data, feature_extractor.config.sample_rate
    )
    with torch.no_grad():
        output = model(audio_feature.unsqueeze(0))
    return output.item()


def interpret_result(output: float) -> str:
    return "Snore" if output > 0.5 else "No Snore"


def main(model_path: str, audio_file: str) -> str:
    ckpt_folder = model_path
    ckpt_path = next(
        (
            os.path.join(ckpt_folder, f)
            for f in os.listdir(ckpt_folder)
            if f.endswith(".pth")
        ),
        None,
    )
    if ckpt_path is None:
        raise FileNotFoundError("No .pth file found in the checkpoint directory.")

    feature_config = FeatureConfig.get_config(
        os.path.join(ckpt_folder, "feature_config.json")
    )
    model_config = ModelConfig.get_config(
        os.path.join(ckpt_folder, "model_config.json")
    )

    model = load_model(ckpt_path, model_config, feature_config)
    feature_extractor = AudioFeatureExtractor()

    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    # Resample if needed, before passing to infer function
    if sample_rate != feature_extractor.config.sample_rate:
        audio_data = librosa.resample(
            audio_data,
            orig_sr=sample_rate,
            target_sr=feature_extractor.config.sample_rate,
        )
        sample_rate = feature_extractor.config.sample_rate

    output = infer(audio_data, sample_rate, model, feature_extractor)
    return interpret_result(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snore Detection Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to the audio file for inference",
    )
    args = parser.parse_args()

    result = main(args.model_path, args.audio_file)
    print(f"Inference result: {result}")
