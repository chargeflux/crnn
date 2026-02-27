import torch

from crnn.data import BLANK_IDX


def greedy(logits: torch.Tensor, vocab: str) -> str:
    if logits.ndim != 2:
        raise ValueError("Expected 2D tensor for logits")
    indices = logits.argmax(-1)
    indices = torch.unique_consecutive(indices)
    indices = indices[indices != BLANK_IDX].cpu()
    res = "".join(vocab[i] for i in indices)
    return res
