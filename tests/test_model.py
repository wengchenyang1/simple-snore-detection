# Copyright (c) 2025, SountIO
import unittest
from unittest.mock import patch

from torch import nn

from src.config import FeatureConfig, ModelConfig
from src.model import SnoreDetectionModel


class TestSnoreDetectionModel(unittest.TestCase):

    def setUp(self):
        self.feature_config_data = {
            "sample_rate": 16000,
            "audio_length_sec": 1,
            "n_mfcc": 40,
            "n_mels": 128,
            "window_size_ms": 30,
            "window_step_ms": 20,
            "method": "mel_spectrogram",
        }
        self.model_config_data = {
            "layers": [
                {
                    "type": "conv2d",
                    "filters": 16,
                    "kernel_size": [3, 3],
                    "activation": "relu",
                },
                {"type": "maxpool2d", "pool_size": [2, 2]},
                {"type": "flatten"},
                {"type": "dense", "units": 10, "activation": "relu"},
                {"type": "dense", "units": 1, "activation": "sigmoid"},
            ]
        }

    @patch("src.config.FeatureConfig.get_config")
    @patch("src.config.ModelConfig.get_config")
    def test_model_structure(self, mock_model_config, mock_feature_config):
        mock_feature_config.return_value = FeatureConfig(self.feature_config_data)
        mock_model_config.return_value = ModelConfig(self.model_config_data)

        model = SnoreDetectionModel(
            mock_model_config.return_value, mock_feature_config.return_value
        )

        # Check the input shape
        self.assertEqual(
            model.input_shape, (1, 128, 51)
        )  # Adjust the expected shape based on your calculation

        # Check the conv_net structure
        conv_layers = list(model.conv_net)
        self.assertIsInstance(conv_layers[0], nn.Conv2d)
        self.assertEqual(conv_layers[0].out_channels, 16)
        self.assertEqual(conv_layers[0].kernel_size, (3, 3))
        self.assertIsInstance(conv_layers[1], nn.ReLU)
        self.assertIsInstance(conv_layers[2], nn.MaxPool2d)
        self.assertEqual(conv_layers[2].kernel_size, (2, 2))

        # Check the dense_net structure
        dense_layers = list(model.dense_net)
        self.assertIsInstance(dense_layers[0], nn.Flatten)
        self.assertIsInstance(dense_layers[1], nn.Linear)
        self.assertEqual(dense_layers[1].out_features, 10)
        self.assertIsInstance(dense_layers[2], nn.ReLU)
        self.assertIsInstance(dense_layers[3], nn.Linear)
        self.assertEqual(dense_layers[3].out_features, 1)
        self.assertIsInstance(dense_layers[4], nn.Sigmoid)


if __name__ == "__main__":
    unittest.main()
