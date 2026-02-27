from typing import Sequence
import pytest
import torch
from crnn.data import (
    BLANK,
    BLANK_IDX,
    DIGITS,
    Vocabulary,
    collate_ctc,
    get_transforms_compose,
)


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        [(1, 28, 28), (1, 32, 32)],
        [(1, 36, 36), (1, 32, 32)],
        [(1, 32, 32), (1, 32, 32)],
        [(1, 36, 72), (1, 32, 64)],
    ],
)
def test_get_transforms_compose_shape(
    input_shape: Sequence[int], output_shape: Sequence[int]
):
    compose = get_transforms_compose()
    input = torch.ones(input_shape)
    output = compose(input)
    assert output.shape == output_shape


def test_get_transforms_compose_dtype():
    compose = get_transforms_compose()
    input = torch.randint(0, 255, (1, 32, 32), dtype=torch.uint8)
    output = compose(input)
    assert output.dtype == torch.float32
    assert output.max() <= 1


def test_get_transforms_compose_grayscale():
    compose = get_transforms_compose()
    input = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
    output = compose(input)
    assert output.size(0) == 1


@pytest.mark.parametrize(
    "batch, expected_image_shape, expected_targets, expected_lengths",
    [
        # Single sample
        (
            [(torch.ones(1, 28, 28), torch.tensor([7]))],
            (1, 1, 28, 28),
            torch.tensor([7]),
            torch.tensor([1]),
        ),
        # Two samples, different lengths
        (
            [
                (torch.ones(1, 28, 28), torch.tensor([8])),
                (torch.ones(1, 28, 28), torch.tensor([5, 3])),
            ],
            (2, 1, 28, 28),
            torch.tensor([8, 5, 3]),
            torch.tensor([1, 2]),
        ),
        # Two samples, same length
        (
            [
                (torch.ones(1, 28, 28), torch.tensor([1, 2])),
                (torch.ones(1, 28, 28), torch.tensor([3, 4])),
            ],
            (2, 1, 28, 28),
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([2, 2]),
        ),
    ],
)
def test_collate_ctc(batch, expected_image_shape, expected_targets, expected_lengths):
    images, targets, lengths = collate_ctc(batch)

    assert images.shape == expected_image_shape

    assert torch.equal(targets, expected_targets)

    assert torch.equal(lengths, expected_lengths)


@pytest.mark.parametrize("vocab, expected_str", [(Vocabulary.DIGITS, BLANK + DIGITS)])
def test_vocabulary(vocab: Vocabulary, expected_str: str):
    assert vocab == expected_str
    assert len(vocab) == len(expected_str)
    assert vocab[BLANK_IDX] == BLANK
    char_to_idx = vocab.char_to_idx()
    assert len(char_to_idx) == len(expected_str)
    assert char_to_idx[expected_str[-1]] == len(expected_str) - 1
