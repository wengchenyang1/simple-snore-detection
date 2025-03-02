# Copyright (c) 2025, SountIO
import json
import os
import shutil
from typing import Any, Dict

CONFIG_PATHS = {
    "feature": "configs/feature_config.json",
    "model": "configs/model_config.json",
    "training": "configs/training_config.json",
}


class FeatureConfig:
    def __init__(self, config: Dict[str, Any]):
        self.sample_rate: int = config["sample_rate"]
        self.audio_length_sec: int = config["audio_length_sec"]
        self.n_mfcc: int = config.get("n_mfcc", 40)
        self.n_mels: int = config.get("n_mels", 128)
        self.window_size_ms: int = config["window_size_ms"]
        self.window_step_ms: int = config["window_step_ms"]
        self.method: str = config.get("method", "mfcc")
        self.sample_width_bytes: int = config.get("sample_width_bytes", 2)

    @classmethod
    def get_config(cls, config_path: str = CONFIG_PATHS["feature"]) -> "FeatureConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config)


class ModelConfig:
    def __init__(self, config: Dict[str, Any]):
        self.layers = config["layers"]

    @classmethod
    def get_config(cls, config_path: str = CONFIG_PATHS["model"]) -> "ModelConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config)


class TrainingConfig:
    def __init__(self, config: Dict[str, Any]):
        self.batch_size: int = config["batch_size"]
        self.num_epochs: int = config["num_epochs"]
        self.learning_rate: float = config["learning_rate"]
        self.num_workers: int = config["num_workers"]

    @classmethod
    def get_config(
        cls, config_path: str = CONFIG_PATHS["training"]
    ) -> "TrainingConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(config)


def copy_config_files(dest_folder: str) -> None:
    for key, path in CONFIG_PATHS.items():
        dest_path = os.path.join(dest_folder, f"{key}_config.json")
        shutil.copy(path, dest_path)
