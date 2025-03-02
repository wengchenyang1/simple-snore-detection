# Snore Detection Project

## Overview

This project aims to detect snoring sounds using a neural network model. The project is designed to be flexible and configurable through JSON configuration files. The main components include feature extraction, model architecture, and training parameters.

## Setup Instructions

1. **Install Dependencies**:
    Ensure you have all the necessary dependencies installed. You can do this by running:

    ```bash
    pip install -r requirements.txt
    ```

2. **Prepare Data**:
    Place your audio data in the `data` folder. Organize the data into `train` and `val` subfolders:

    ```
    data/
    ├── train/
    │   ├── snore/
    │   └── no_snore/
    └── val/
        ├── snore/
        └── no_snore/
    ```

3. **Configure Settings**:
    Set up the feature extraction, model architecture, and training parameters in the following configuration files:

    ```
    configs/feature_config.json
    configs/model_config.json
    configs/training_config.json
    ```

## Model Setup

The model is configured using the `ModelConfig` and `FeatureConfig` classes. The `SnoreDetectionModel` class defines the neural network architecture based on the configurations provided.

## Training the Model

To train the model, run the following command:

```bash
python -m src.train
```

This will use the configurations specified in the JSON files to train the model on the data in the `data/train` folder and validate it using the data in the `data/val` folder.

## Inference on Audio Files

To perform inference on a specific audio file, run:

```Bash
python -m src.inference --model_path <path_to_model_checkpoint> --audio_file <path_to_audio_file>
```

Replace <path_to_model_checkpoint> with the path to your model checkpoint directory and <path_to_audio_file> with the path to the audio file you want to analyze.

## Real-Time Inference

To perform real-time inference using the microphone, run:

```Bash
python -m src.inference --model_path <path_to_model_checkpoint> --realtime
```

Replace <path_to_model_checkpoint> with the path to your model checkpoint directory.

## Run unit tests

```
python -m unittest discover -s tests
```
