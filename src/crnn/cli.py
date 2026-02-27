from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import logging
from pathlib import Path
import sys

import torch

from crnn import infer, train
from crnn.data import DataSplit, Vocabulary, load_mnist
from crnn.log import configure_logging
from crnn.net import CRNN


class Dataset(StrEnum):
    MNIST = "MNIST"

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
    vocabulary: str


@dataclass
class TrainConfig(Config):
    dataset: Dataset
    output_dir: Path

    @classmethod
    def from_args(cls, parsed_args: Namespace) -> "TrainConfig":
        return cls(
            command=parsed_args.command,
            device=parsed_args.device,
            vocabulary=parsed_args.vocabulary,
            dataset=parsed_args.dataset,
            output_dir=parsed_args.output_dir
            or Path(
                f"experiments/{parsed_args.dataset}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
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
        help="specify string representing character set for decoding with a blank token character at the beginning",
        default=Vocabulary.DIGITS,
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
        "-o",
        "--output_dir",
        type=Path,
        help="directory to store the model and config. Defaults to experiments/<dataset>/Ymd_HMS",
    )

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
    configure_logging()

    config = parse_args(sys.argv[1:])
    logging.debug(f"Parsed args: {config}")

    vocab = config.vocabulary
    logging.info(f"Using vocabulary: {vocab}")
    device = config.device

    if isinstance(config, TrainConfig):
        hparams = train.Hyperparameters(epochs=3)
        logging.info(f"Hyperparameters: {hparams}")

        if config.dataset == Dataset.MNIST:
            logging.info("Loading MNIST dataset")
            train_loader = load_mnist(hparams.batch_size, DataSplit.TRAIN, hparams.seed)
            val_loader = load_mnist(hparams.batch_size, DataSplit.TRAIN, hparams.seed)
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset.value}")

        logging.info(f"Output directory: {config.output_dir}")
        train.train(
            hparams,
            Vocabulary(vocab),
            device,
            train_loader,
            val_loader,
            config.output_dir,
        )
    elif isinstance(config, InferConfig):
        model = CRNN(len(vocab))
        model.load_state_dict(
            torch.load(config.model_path, weights_only=True, map_location=device)
        )
        model.to(device)

        print(infer.predict(config.file, device, vocab, model))


if __name__ == "__main__":
    main()
