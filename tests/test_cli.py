from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from crnn.cli import Dataset, InferConfig, TrainConfig, parse_args
from crnn.train import Hyperparameters


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            ["train", "--device", "cuda", "-o", "experiments"],
            TrainConfig(
                "train",
                "cuda",
                None,
                "INFO",
                Dataset.MNIST,
                None,
                Path("experiments"),
                Hyperparameters(),
            ),
        ),
        (
            [
                "train",
                "--device",
                "cuda",
                "--dataset",
                "LABEL_FILE",
                "-i",
                "dataset",
                "-o",
                "experiments",
            ],
            TrainConfig(
                "train",
                "cuda",
                None,
                "INFO",
                Dataset.LABEL_FILE,
                Path("dataset"),
                Path("experiments"),
                Hyperparameters(),
            ),
        ),
        (
            [
                "train",
                "--device",
                "cuda",
                "--dataset",
                "LABEL_FILE",
                "-i",
                "dataset",
                "-o",
                "experiments",
                "--vocabulary",
                "-0123456789",
            ],
            TrainConfig(
                "train",
                "cuda",
                "-0123456789",
                "INFO",
                Dataset.LABEL_FILE,
                Path("dataset"),
                Path("experiments"),
                Hyperparameters(),
            ),
        ),
        (
            ["train", "--device", "cuda", "--epochs", "3", "-o", "experiments"],
            TrainConfig(
                "train",
                "cuda",
                None,
                "INFO",
                Dataset.MNIST,
                None,
                Path("experiments"),
                Hyperparameters(epochs=3),
            ),
        ),
        (
            [
                "infer",
                "--device",
                "cuda",
                "-f",
                "IMAGE",
                "-m",
                "path/to/model",
            ],
            InferConfig(
                "infer",
                "cuda",
                None,
                "INFO",
                Path("path/to/model"),
                Path("IMAGE"),
            ),
        ),
        (
            [
                "infer",
                "--device",
                "cuda",
                "-f",
                "IMAGE",
                "-m",
                "path/to/model",
                "--vocabulary",
                "-0123456789",
            ],
            InferConfig(
                "infer",
                "cuda",
                "-0123456789",
                "INFO",
                Path("path/to/model"),
                Path("IMAGE"),
            ),
        ),
    ],
)
def test_parse_args(input: List[str], expected):
    assert parse_args(input) == expected
