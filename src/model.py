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

        # Load the config file
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.feature_config = self.config["feature_extraction"]
        self.model_config = self.config["model"]

        # Calculate expected MFCC input shape
        sample_rate = self.feature_config["sample_rate"]
        audio_length = self.feature_config["audio_length"]
        window_size_ms = self.feature_config["window_size_ms"]
        window_step_ms = self.feature_config["window_step_ms"]
        n_mfcc = self.feature_config["n_mfcc"]

        window_size_samples = int(sample_rate * window_size_ms / 1000)
        window_step_samples = int(sample_rate * window_step_ms / 1000)
        clip_length_samples = int(sample_rate * audio_length)
        num_frames = (
            clip_length_samples - window_size_samples
        ) // window_step_samples + 1
        self.input_shape = (1, n_mfcc, num_frames)  # (channels, height, width)

        conv_layers = []
        in_channels = 1  # MFCC input has 1 channel
        for layer in self.model_config["layers"]:
            if layer["type"] == "conv2d":
                conv_layer = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=layer["filters"],
                    kernel_size=tuple(layer["kernel_size"]),
                    stride=1,
                    padding=1,  # Padding to preserve spatial dimensions initially
                )
                conv_layers.append(conv_layer)
                in_channels = layer["filters"]
                if "activation" in layer:
                    conv_layers.append(self.get_activation(layer["activation"]))
            elif layer["type"] == "maxpool2d":
                pool_layer = nn.MaxPool2d(kernel_size=tuple(layer["pool_size"]))
                conv_layers.append(pool_layer)
            elif layer["type"] == "flatten":
                break  # Stop before flatten

        self.conv_net = nn.Sequential(*conv_layers)

        # Compute the flattened size after conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)  # Batch size of 1
            dummy_output = self.conv_net(dummy_input)
            flattened_size = dummy_output.view(dummy_output.size(0), -1).size(1)

        # Build dense layers
        dense_layers = []
        in_features = flattened_size
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
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input MFCC features with shape (batch, 1, n_mfcc, num_frames).

        Returns:
            torch.Tensor: Model output (e.g., probability of snoring).
        """
        x = self.conv_net(x)
        x = self.dense_net(x)
        return x

    def get_activation(self, activation):
        """
        Return the specified activation function.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.Module: PyTorch activation module.
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
