import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    Adapted CRNN model described by the paper
    "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition" by Shi et al.

    Input images must be grayscale with a height of 32.
    """

    def __init__(self, num_outputs: int):
        super().__init__()

        self.cnn = nn.Sequential(
            # layer 1
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(True),
            # layer 2
            nn.MaxPool2d(2, 2),
            # layer 3
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            # layer 4
            nn.MaxPool2d(2, 2),
            # layer 5
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            # layer 6
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            # layer 7
            # Paper has a typo for asymmetric max pooling layer.
            # It should be 1 x 2 (W x H) stride sizes
            # https://github.com/bgshih/crnn/issues/6
            nn.MaxPool2d(2, (2, 1), (0, 1)),
            # layer 8 + 9
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # layer 10 + 11
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # layer 12
            nn.MaxPool2d(2, (2, 1), (0, 1)),
            # layer 13
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True),
        )

        # LSTM -> LSTM -> Linear
        # Differs from paper which does LSTM -> Linear -> LSTM -> Linear
        self.rnn = nn.LSTM(512, 256, 2, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(512, num_outputs)

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError("Input shape should be Batch x Channel x Height x Width")
        if x.size(1) != 1:
            raise ValueError("Image should be grayscale")
        if x.size(2) != 32 != 0:
            raise ValueError("Height should be 32")

        cnn_out: torch.Tensor = self.cnn(x)

        B, C, H, W = cnn_out.size()
        assert H == 1, "Height should be 1"

        # B x C x H x W -> B x C x W -> W x B x C
        rnn_in = cnn_out.view(B, C * H, W).permute(2, 0, 1)

        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(rnn_in)

        return self.fc(rnn_out)
