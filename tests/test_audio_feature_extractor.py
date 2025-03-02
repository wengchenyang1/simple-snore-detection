# Copyright (c) 2025, SountIO
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.audiofeature import AudioFeatureExtractor


class TestAudioFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.mock_config = {
            "sample_rate": 16000,
            "audio_length_sec": 1,
            "n_mfcc": 40,
            "n_mels": 128,
            "window_size_ms": 30,
            "window_step_ms": 20,
            "method": "mfcc",
        }
        self.patcher = patch(
            "src.config.FeatureConfig.get_config",
            return_value=MagicMock(**self.mock_config),
        )
        self.mock_get_config = self.patcher.start()
        self.extractor = AudioFeatureExtractor()

    def tearDown(self):
        self.patcher.stop()

    def _calculate_expected_frames(
        self, sample_rate, audio_length_sec, window_size_ms, window_step_ms
    ):
        window_size_samples = int(sample_rate * window_size_ms / 1000)
        window_step_samples = int(sample_rate * window_step_ms / 1000)
        clip_length_samples = int(sample_rate * audio_length_sec)
        expected_frames = (
            clip_length_samples - window_size_samples
        ) // window_step_samples + 3
        return expected_frames

    def _generate_audio_data(self, sample_rate, audio_length_sec):
        # Generate a sine wave as test audio data
        t = np.linspace(
            0, audio_length_sec, int(sample_rate * audio_length_sec), endpoint=False
        )
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        return audio_data

    def _test_get_feature(self, method, from_file=True):
        self.mock_config["method"] = method
        self.mock_get_config.return_value = MagicMock(
            **self.mock_config
        )  # Update the mock config
        self.extractor = (
            AudioFeatureExtractor()
        )  # Recreate the extractor with the updated method
        sample_rate = self.mock_config["sample_rate"]
        audio_length_sec = self.mock_config["audio_length_sec"]
        audio_data = self._generate_audio_data(sample_rate, audio_length_sec)
        audio_path = "path/to/sample_audio.wav"

        if from_file:
            with patch("librosa.load", return_value=(audio_data, sample_rate)):
                features = self.extractor.get_feature_from_file(audio_path)
        else:
            features = self.extractor.get_feature_from_stream(audio_data, sample_rate)

        self.assertIsInstance(features, torch.Tensor)
        if method == "mfcc":
            self.assertEqual(features.shape[1], self.mock_config["n_mfcc"])
        else:
            self.assertEqual(features.shape[1], self.mock_config["n_mels"])

        expected_frames = self._calculate_expected_frames(
            sample_rate, audio_length_sec, 30, 20
        )
        self.assertEqual(features.shape[2], expected_frames)

    def test_get_feature_from_stream_mfcc(self):
        self._test_get_feature("mfcc", from_file=False)

    def test_get_feature_from_file_mfcc(self):
        self._test_get_feature("mfcc", from_file=True)

    def test_get_feature_from_stream_mel_spectrogram(self):
        self._test_get_feature("mel_spectrogram", from_file=False)

    def test_get_feature_from_file_mel_spectrogram(self):
        self._test_get_feature("mel_spectrogram", from_file=True)

    def test_get_feature_from_stream_with_different_sample_rate_and_length(self):
        self.mock_config["method"] = "mfcc"
        self.mock_get_config.return_value = MagicMock(**self.mock_config)
        self.extractor = AudioFeatureExtractor()

        different_sample_rate = 22050
        different_audio_length_sec = 2
        audio_data = self._generate_audio_data(
            different_sample_rate, different_audio_length_sec
        )

        features = self.extractor.get_feature_from_stream(
            audio_data, different_sample_rate
        )
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[1], self.mock_config["n_mfcc"])

        expected_frames = self._calculate_expected_frames(
            self.mock_config["sample_rate"],
            self.mock_config["audio_length_sec"],
            self.mock_config["window_size_ms"],
            self.mock_config["window_step_ms"],
        )
        self.assertEqual(features.shape[2], expected_frames)

    def test_get_feature_from_file_with_different_sample_rate_and_length(self):
        self.mock_config["method"] = "mfcc"
        self.mock_get_config.return_value = MagicMock(**self.mock_config)
        self.extractor = AudioFeatureExtractor()

        different_sample_rate = 22050
        different_audio_length_sec = 2
        audio_data = self._generate_audio_data(
            different_sample_rate, different_audio_length_sec
        )

        # Calculate expected frames based on config (not the different sample rate/length)
        expected_frames = self._calculate_expected_frames(
            self.mock_config["sample_rate"],
            self.mock_config["audio_length_sec"],
            self.mock_config["window_size_ms"],
            self.mock_config["window_step_ms"],
        )

        audio_path = "path/to/sample_audio.wav"
        with patch("librosa.load", return_value=(audio_data, different_sample_rate)):
            features = self.extractor.get_feature_from_file(audio_path)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[1], self.mock_config["n_mfcc"])
        self.assertEqual(features.shape[2], expected_frames)

    def test_get_feature_from_file_with_short_audio(self):
        self.mock_config["method"] = "mfcc"
        self.mock_get_config.return_value = MagicMock(**self.mock_config)
        self.extractor = AudioFeatureExtractor()

        short_audio_length_sec = 0.5
        audio_data = self._generate_audio_data(
            self.mock_config["sample_rate"], short_audio_length_sec
        )

        audio_path = "path/to/sample_audio.wav"
        with patch(
            "librosa.load", return_value=(audio_data, self.mock_config["sample_rate"])
        ):
            with self.assertRaises(ValueError):
                self.extractor.get_feature_from_file(audio_path)


if __name__ == "__main__":
    unittest.main()
