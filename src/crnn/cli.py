from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import logging
from pathlib import Path
import sys
from typing import Optional

import torch

from crnn import infer, train
from crnn.data import DataSplit, Vocabulary, load_labelfile_dataset, load_mnist
from crnn.log import configure_logging


class Dataset(StrEnum):
    MNIST = "MNIST"
    LABEL_FILE = "LABEL_FILE"

    def get_vocabulary(self) -> Vocabulary:
        match self:
            case Dataset.MNIST:
                return Vocabulary.DIGITS
            case _:
                raise ValueError(f"No vocabulary defined for {self}")


@dataclass
class Config:
    command: str
    device: str
    vocabulary: Optional[str]
    logging_level: str


@dataclass
class TrainConfig(Config):
    dataset: Dataset
    input_dir: Optional[Path]
    output_dir: Path
    hp: train.Hyperparameters

    @classmethod
    def from_args(cls, parsed_args: Namespace) -> "TrainConfig":
        return cls(
            command=parsed_args.command,
            device=parsed_args.device,
            vocabulary=parsed_args.vocabulary,
            logging_level=parsed_args.logging_level,
            dataset=parsed_args.dataset,
            input_dir=parsed_args.input_dir,
            output_dir=parsed_args.output_dir
            or Path(
                f"experiments/{parsed_args.dataset.lower()}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
            hp=train.Hyperparameters.from_args(parsed_args),
        )


@dataclass
class InferConfig(Config):
    model_path: Path
    file: Path

    @classmethod
    def from_args(cls, parsed_args: Namespace) -> "InferConfig":
        return cls(
            command=parsed_args.command,
            device=parsed_args.device,
            vocabulary=parsed_args.vocabulary,
            logging_level=parsed_args.logging_level,
            model_path=Path(parsed_args.model_path),
            file=Path(parsed_args.file),
        )


def parse_args(args) -> TrainConfig | InferConfig:
    root_parser = ArgumentParser(add_help=False)
    root_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run on (cuda or cpu)",
    )
    root_parser.add_argument(
        "--vocabulary",
        type=str,
        help="specify string representing the character set for decoding with a blank token character at the beginning",
    )
    root_parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
        type=str,
        help="set logging level",
    )

    parser = ArgumentParser("crnn", description="Train and infer from a CRNN model")
    subparsers = parser.add_subparsers(
        dest="command", help="available commands", required=True
    )
    train_parser = subparsers.add_parser(
        "train", parents=[root_parser], description="Train a CRNN model"
    )
    train_parser.add_argument(
        "--dataset",
        default=Dataset.MNIST,
        choices=[d.value for d in Dataset],
        help=f"datasets: {[d.value for d in Dataset]}",
    )
    train_parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        help="directory containing 'train' and 'test' folders. Required for LABEL_FILE dataset where each image has a corresponding text file containing the label",
    )
    train_parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="directory to store the model and config. Defaults to experiments/<dataset>/Ymd_HMS",
    )
    train.Hyperparameters.to_args(train_parser)

    infer_parser = subparsers.add_parser(
        "infer",
        parents=[root_parser],
        description="Infer text from image using CRNN model",
    )
    infer_parser.add_argument(
        "-f", "--file", type=Path, help="image file to run inference on", required=True
    )
    infer_parser.add_argument(
        "-m", "--model-path", type=Path, help="path to model", required=True
    )

    parsed = parser.parse_args(args)

    match parsed.command:
        case "train":
            return TrainConfig.from_args(parsed)
        case "infer":
            return InferConfig.from_args(parsed)
        case _:
            raise ValueError("Unknown command")


def main():
    config = parse_args(sys.argv[1:])
    configure_logging(config.logging_level)
    logging.debug(f"Parsed args: {config}")

    vocab = config.vocabulary
    if vocab is not None:
        logging.info(f"Using vocabulary: {vocab}")
    device = config.device

    if isinstance(config, TrainConfig):
        hparams = config.hp
        logging.info(f"Hyperparameters: {hparams}")

        if config.dataset == Dataset.MNIST:
            logging.info("Loading MNIST dataset")
            train_loader = load_mnist(
                hparams.batch_size,
                DataSplit.TRAIN,
                hparams.seed,
                hparams.validation_split,
            )
            val_loader = load_mnist(
                hparams.batch_size,
                DataSplit.TRAIN,
                hparams.seed,
                hparams.validation_split,
            )
            if vocab is None:
                vocab = config.dataset.get_vocabulary()
                logging.info(f"Defaulting vocabulary to digits: {vocab}")
        elif config.dataset == Dataset.LABEL_FILE:
            logging.info("Loading label file dataset")
            if vocab is None:
                raise ValueError("Vocabulary must be specified for label file dataset")
            if config.input_dir is None or not config.input_dir.exists():
                raise ValueError(
                    f"Invalid dataset path: {config.input_dir}. Specify path to directory with 'train' and 'test' subdirectories for {config.dataset} dataset"
                )

            char_to_idx = {char: i for i, char in enumerate(vocab)}
            train_loader = load_labelfile_dataset(
                config.input_dir / "train",
                char_to_idx,
                hparams.batch_size,
                DataSplit.TRAIN,
                hparams.seed,
                hparams.validation_split,
            )
            val_loader = load_labelfile_dataset(
                config.input_dir / "train",
                char_to_idx,
                hparams.batch_size,
                DataSplit.VALIDATION,
                hparams.seed,
                hparams.validation_split,
            )
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset.value}")

        logging.info(f"Output directory: {config.output_dir}")
        train.train(
            hparams,
            vocab,
            device,
            train_loader,
            val_loader,
            config.output_dir,
        )
    elif isinstance(config, InferConfig):
        logging.info(
            f"Running inference on {config.file} with model {config.model_path}"
        )
        model, vocab = train.load_model(config.model_path, device, vocab)

        result = infer.predict(config.file, device, vocab, model)
        print(result)


if __name__ == "__main__":
    main()
