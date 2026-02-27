from dataclasses import asdict, dataclass
from enum import StrEnum
import json
import logging
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader

from crnn.decoder import greedy
from crnn.data import Vocabulary
from crnn.net import CRNN

logger = logging.getLogger(__name__)


class Optimizer(StrEnum):
    SGD = "sgd"
    ADADELTA = "adadelta"

    def create(
        self, params: Iterator[torch.nn.Parameter], lr: float
    ) -> torch.optim.Optimizer:
        match self:
            case self.SGD:
                return torch.optim.SGD(params, lr)
            case self.ADADELTA:
                return torch.optim.Adadelta(params)


@dataclass
class Hyperparameters:
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 1e-4
    optimizer: Optimizer = Optimizer.ADADELTA
    seed: int = 0


def save_model(hparams: Hyperparameters, model: nn.Module, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = output_dir / "model.pth"
    hparams_save_path = output_dir / "model.json"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Saved model: {model_save_path}")
    with open(hparams_save_path, "w") as f:
        json.dump(asdict(hparams), f, indent=4)
    logging.info(f"Saved model config: {hparams_save_path}")


def train(
    hparams: Hyperparameters,
    vocab: Vocabulary,
    device: str,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    output_dir: Path,
):
    torch.manual_seed(hparams.seed)

    model = CRNN(len(vocab)).to(device)
    optimizer = hparams.optimizer.create(model.parameters(), hparams.learning_rate)
    criterion = torch.nn.CTCLoss().to(device)

    for epoch in range(hparams.epochs):
        train_loss = 0.0
        model.train()

        for batch_idx, (X, y, y_lens) in enumerate(train_loader):
            X, y, y_lens = X.to(device), y.to(device), y_lens.to(device)

            logits = model(X)
            T, N, _ = logits.shape

            input_lengths = torch.full(
                size=(N,), fill_value=T, device=device, dtype=torch.long
            )

            loss = criterion(
                logits.log_softmax(2),
                y,
                input_lengths,
                y_lens,
            )

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                epoch_batch = f"Epoch: {epoch + 1:03d}/{hparams.epochs:03d} | Batch: {batch_idx:03d}/{len(train_loader):03d}"
                logger.info(f"{epoch_batch} | Loss: {loss:.4f}")
                if logger.isEnabledFor(logging.DEBUG):
                    debug_logits = logits[:, 0, :]
                    debug_predicted = greedy(debug_logits, vocab)
                    debug_y = y[: y_lens[0]].cpu()
                    debug_expected = "".join(vocab[i] for i in debug_y)
                    logger.debug(
                        f"{epoch_batch} | Predicted: {debug_predicted} | Expected: {debug_expected}",
                    )

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, y, y_lens in valid_loader:
                X, y, y_lens = X.to(device), y.to(device), y_lens.to(device)

                logits = model(X)
                T, N, _ = logits.shape

                input_lengths = torch.full(size=(N,), fill_value=T, device=device)

                loss = criterion(
                    logits.log_softmax(2),
                    y,
                    input_lengths,
                    y_lens,
                )

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        logger.info(
            f"Epoch: {epoch + 1:03d}/{hparams.epochs:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    save_model(hparams, model, output_dir)
