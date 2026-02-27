from enum import StrEnum
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import v2

BLANK = "-"
BLANK_IDX = 0
DIGITS = "0123456789"


class BaseVocabulary(StrEnum):
    def __new__(cls, value):
        member = str.__new__(cls, BLANK + value)
        member._value_ = BLANK + value
        return member


class Vocabulary(BaseVocabulary):
    DIGITS = DIGITS

    def __len__(self):
        return len(self.value)

    def char_to_idx(self):
        return {char: i for i, char in enumerate(self.value)}


class DataSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def get_transforms_compose() -> v2.Compose:
    return v2.Compose(
        [
            v2.Grayscale(),
            v2.Resize(32),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def collate_ctc(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, targets = zip(*batch)

    images = torch.stack(images)
    targets_concat = torch.cat(targets)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)

    return images, targets_concat, target_lengths


def load_mnist(
    batch_size: int,
    split: DataSplit,
    seed: int,
    path: str = "data/mnist",
    val_split: float = 0.2,
) -> DataLoader:
    is_test = split == DataSplit.TEST
    dataset = datasets.MNIST(
        root=path,
        train=not is_test,
        download=True,
        transform=get_transforms_compose(),
        target_transform=lambda y: torch.tensor(
            [y + 1], dtype=torch.long
        ),  # targets are 0-9 but 0 is blank token for CTCLoss
    )

    if not is_test:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        dataset = train_ds if (split == DataSplit.TRAIN) else val_ds

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == DataSplit.TRAIN),
        collate_fn=collate_ctc,
    )


class LabelFileDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        char_to_idx: Dict[str, int],
        transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform

        extensions = {".png", ".jpg", ".jpeg"}
        self.image_paths = [
            f for f in root_dir.glob("**/*") if f.suffix.lower() in extensions
        ]
        self.char_to_idx = char_to_idx

    def __getitem__(self, index):
        image_path: Path = self.image_paths[index]

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        text_path = image_path.with_suffix(".txt")

        if text_path.exists():
            with open(text_path, "r") as f:
                label_str = f.read().strip()
        else:
            raise ValueError(f"Failed to find label for image {image_path}")

        label_indices = [self.char_to_idx[c] for c in label_str]

        return image, torch.LongTensor(label_indices)

    def __len__(self):
        return len(self.image_paths)


def load_labelfile_dataset(
    path: Path,
    vocab: Vocabulary,
    batch_size: int,
    split: DataSplit,
    seed: int,
    val_split: float = 0.2,
):
    is_test = split == DataSplit.TEST

    dataset = LabelFileDataset(
        path,
        vocab.char_to_idx(),
        transform=get_transforms_compose(),
    )

    if not is_test:
        dataset = LabelFileDataset(
            path,
            vocab.char_to_idx(),
            transform=get_transforms_compose(),
        )
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        dataset = train_ds if (split == DataSplit.TRAIN) else val_ds

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == DataSplit.TRAIN),
        collate_fn=collate_ctc,
    )
