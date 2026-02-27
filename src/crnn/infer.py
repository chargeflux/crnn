from pathlib import Path

from PIL import Image

from crnn.data import get_transforms_compose
from crnn.decoder import greedy
from crnn.net import CRNN


def predict(filename: Path, device: str, vocab: str, model: CRNN):
    img = Image.open(filename)
    compose = get_transforms_compose()
    img = compose(img).to(device)
    logits = model(img.unsqueeze(0))
    return greedy(logits[:, 0, :], vocab)
