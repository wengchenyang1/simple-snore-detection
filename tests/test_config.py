# Copyright (c) 2025, SountIO
import json
import os
import shutil
import unittest
from unittest.mock import mock_open, patch

from src.config import (
    CONFIG_PATHS,
    FeatureConfig,
    ModelConfig,
    TrainingConfig,
    copy_config_files,
)


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.feature_config_data = {
            "sample_rate": 16000,
            "audio_length_sec": 1,
            "n_mfcc": 40,
            "n_mels": 128,
            "window_size_ms": 30,
            "window_step_ms": 20,
            "method": "mfcc",
        }
        self.model_config_data = {
            "layers": [
                {
                    "type": "conv2d",
                    "filters": 32,
                    "kernel_size": [3, 3],
                    "activation": "relu",
                },
                {"type": "maxpool2d", "pool_size": [2, 2]},
                {"type": "flatten"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 10, "activation": "softmax"},
            ]
        }
        self.training_config_data = {
            "batch_size": 32,
            "num_epochs": 10,
            "learning_rate": 0.001,
            "num_workers": 4,
        }

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(
            {
                "sample_rate": 16000,
                "audio_length_sec": 1,
                "n_mfcc": 40,
                "n_mels": 128,
                "window_size_ms": 30,
                "window_step_ms": 20,
                "method": "mfcc",
            }
        ),
    )
    def test_feature_config(self, mock_file):
        config = FeatureConfig.get_config()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.audio_length_sec, 1)
        self.assertEqual(config.n_mfcc, 40)
        self.assertEqual(config.n_mels, 128)
        self.assertEqual(config.window_size_ms, 30)
        self.assertEqual(config.window_step_ms, 20)
        self.assertEqual(config.method, "mfcc")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(
            {
                "layers": [
                    {
                        "type": "conv2d",
                        "filters": 32,
                        "kernel_size": [3, 3],
                        "activation": "relu",
                    },
                    {"type": "maxpool2d", "pool_size": [2, 2]},
                    {"type": "flatten"},
                    {"type": "dense", "units": 128, "activation": "relu"},
                    {"type": "dense", "units": 10, "activation": "softmax"},
                ]
            }
        ),
    )
    def test_model_config(self, mock_file):
        config = ModelConfig.get_config()
        self.assertEqual(len(config.layers), 5)
        self.assertEqual(config.layers[0]["type"], "conv2d")
        self.assertEqual(config.layers[0]["filters"], 32)
        self.assertEqual(config.layers[0]["kernel_size"], [3, 3])
        self.assertEqual(config.layers[0]["activation"], "relu")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(
            {
                "batch_size": 32,
                "num_epochs": 10,
                "learning_rate": 0.001,
                "num_workers": 4,
            }
        ),
    )
    def test_training_config(self, mock_file):
        config = TrainingConfig.get_config()
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.num_epochs, 10)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.num_workers, 4)

    @patch("shutil.copy")
    def test_copy_config_files(self, mock_copy):
        dest_folder = "test_dest"
        copy_config_files(dest_folder)
        for key, path in CONFIG_PATHS.items():
            dest_path = os.path.join(dest_folder, f"{key}_config.json")
            mock_copy.assert_any_call(path, dest_path)


if __name__ == "__main__":
    unittest.main()
