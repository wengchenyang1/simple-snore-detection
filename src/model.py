# Copyright (c) 2025, SountIO
import json

import torch
from torch import nn


class SnoreDetectionModel(nn.Module):
    def __init__(self, config_path):
        """
        Initialize the SnoreDetectionModel with a config file.

        Args:
            config_path (str): Path to the JSON config file.
        """
        super().__init__()
        self._load_config(config_path)
        self._compute_input_shape()
        self._build_conv_network()
        self._compute_flattened_size()
        self._build_dense_network()

    def _load_config(self, config_path):
        """Load the JSON config file and set feature and model configurations."""
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.feature_config = self.config["feature_extraction"]
        self.model_config = self.config["model"]

    def _compute_input_shape(self):
        """Calculate the input shape based on feature extraction parameters."""
        sample_rate = self.feature_config["sample_rate"]
        audio_length_sec = self.feature_config["audio_length_sec"]
        window_size_ms = self.feature_config["window_size_ms"]
        window_step_ms = self.feature_config["window_step_ms"]
        n_mfcc = self.feature_config["n_mfcc"]

        window_size_samples = int(sample_rate * window_size_ms / 1000)
        window_step_samples = int(sample_rate * window_step_ms / 1000)
        clip_length_samples = int(sample_rate * audio_length_sec)
        num_frames = (
            clip_length_samples - window_size_samples
        ) // window_step_samples + 1
        self.input_shape = (1, n_mfcc, num_frames)

    def _build_conv_network(self):
        """Construct the convolutional layers up to the flatten operation."""
        conv_layers = []
        in_channels = 1
        for layer in self.model_config["layers"]:
            if layer["type"] == "conv2d":
                conv_layer = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer["filters"],
                    kernel_size=tuple(layer["kernel_size"]),
                    stride=1,
                    padding=1,
                )
                conv_layers.append(conv_layer)
                in_channels = layer["filters"]
                if "activation" in layer:
                    conv_layers.append(self.get_activation(layer["activation"]))
            elif layer["type"] == "maxpool2d":
                pool_layer = nn.MaxPool2d(kernel_size=tuple(layer["pool_size"]))
                conv_layers.append(pool_layer)
            elif layer["type"] == "flatten":
                break
        self.conv_net = nn.Sequential(*conv_layers)

    def _compute_flattened_size(self):
        """Calculate the size of the flattened output from the conv network."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            dummy_output = self.conv_net(dummy_input)
            self.flattened_size = dummy_output.view(dummy_output.size(0), -1).size(1)

    def _build_dense_network(self):
        """Construct the dense layers after the flatten operation."""
        dense_layers = []
        in_features = self.flattened_size
        for layer in self.model_config["layers"]:
            if layer["type"] == "flatten":
                dense_layers.append(nn.Flatten())
            elif layer["type"] == "dense":
                dense_layer = nn.Linear(in_features, layer["units"])
                dense_layers.append(dense_layer)
                in_features = layer["units"]
                if "activation" in layer:
                    dense_layers.append(self.get_activation(layer["activation"]))
        self.dense_net = nn.Sequential(*dense_layers)

    def forward(self, x):
        """Define the forward pass through the network."""
        x = self.conv_net(x)
        x = self.dense_net(x)
        return x

    def get_activation(self, activation):
        """Return the specified activation function."""
        if activation == "relu":
            return nn.ReLU()

        if activation == "sigmoid":
            return nn.Sigmoid()

        raise ValueError(f"Unsupported activation: {activation}")

    def print_model(self):
        """Print all model layers in sequential order."""
        print("Model Structure:")
        all_layers = list(self.conv_net) + list(self.dense_net)
        for i, layer in enumerate(all_layers):
            print(f"Layer {i}: {layer}")


if __name__ == "__main__":
    CONFIG_PATH = "src/config.json"

    model = SnoreDetectionModel(CONFIG_PATH)
    print("\nInput Shape:", model.input_shape)
    model.print_model()
