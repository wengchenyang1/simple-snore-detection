# Copyright (c) 2025, SountIO
import argparse
import os
import time as tm
from datetime import datetime

import librosa
import numpy as np
import sounddevice as sd
import torch
from torch import nn

from src.audiofeature import AudioFeatureExtractor
from src.config import FeatureConfig, ModelConfig
from src.model import SnoreDetectionModel


def list_audio_devices():
    """Lists all available audio devices."""
    devices = sd.query_devices()
    print("Available audio devices:")
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")


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
    threshold = 0.6
    result = "Snore" if output > threshold else "No Snore"
    color_code = "\033[91m" if result == "Snore" else "\033[92m"
    reset_code = "\033[0m"
    return f"{color_code}{result} (Score: {output:.2f}, Threshold: {threshold}){reset_code}"


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


def realtime_inference(model_path: str, audio_device_index: int):
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

    device_info = sd.query_devices(audio_device_index)
    samplerate = int(device_info["default_samplerate"])
    channels = 1
    dtype = np.float32  # ensure float32, to match librosa.

    print(
        f"Recording from device {audio_device_index} with samplerate: {samplerate} Hz"
    )

    previous_result = None
    buffer = []
    buffer_size = feature_config.sample_rate * feature_config.audio_length_sec
    inference_in_progress = False  # Add a flag to indicate inference status

    def callback(indata, frames, time, status):
        nonlocal previous_result, inference_in_progress
        if status:
            print(status)
        audio_data = indata.flatten()
        if samplerate != feature_config.sample_rate:
            audio_data = librosa.resample(
                audio_data, orig_sr=samplerate, target_sr=feature_config.sample_rate
            )

        if not inference_in_progress:  # Only add to buffer if not inferring.
            buffer.extend(audio_data)

        if len(buffer) >= buffer_size and not inference_in_progress:
            inference_in_progress = True  # set the flag.
            audio_chunk = np.array(buffer[:buffer_size])
            buffer[:] = buffer[buffer_size:]  # clear the buffer.
            output = infer(
                audio_chunk, feature_config.sample_rate, model, feature_extractor
            )
            result = interpret_result(output)

            if result != previous_result:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # get current time.
                print(f"[{now}] {result}")
                previous_result = result
            inference_in_progress = False  # reset the flag.

    try:
        with sd.InputStream(
            device=audio_device_index,
            channels=channels,
            samplerate=samplerate,
            dtype=dtype,
            callback=callback,
        ):
            print("Press Ctrl+C to stop recording...")
            while True:
                tm.sleep(0.1)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    except Exception as e:
        print(f"Error during realtime inference: {e}")


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
        help="Path to the audio file for inference",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Enable realtime inference from microphone",
    )

    args = parser.parse_args()

    if args.realtime:
        list_audio_devices()
        try:
            device_index = int(input("Enter the device index for realtime recording: "))
            realtime_inference(args.model_path, device_index)
        except ValueError:
            print("Invalid device index.")
    elif args.audio_file:
        INFERENCE_RESULT = main(args.model_path, args.audio_file)
        print(f"Inference result: {INFERENCE_RESULT}")
    else:
        print("Please provide either --audio_file or --realtime.")
