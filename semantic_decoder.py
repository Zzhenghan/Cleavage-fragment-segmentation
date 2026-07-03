"""
Semantic decoder for the fragment branch.
Input [B, 256, 64, 64] -> output [B, num_classes, 1024, 1024].
For num_classes=2, channel 0 is background and channel 1 is fragment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvReLU(in_channels, out_channels, 3)
        self.conv2 = ConvReLU(out_channels, out_channels, 3)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SemanticDecoder(nn.Module):
    """
    Input:  [B, 256, 64, 64]
    Output: [B, num_classes, 1024, 1024]

    When num_classes=2:
        channel 0 = background
        channel 1 = fragment
        Prediction: argmax(softmax(out, dim=1)) gives {0, 1}.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.stem = ConvReLU(256, 128, 3)
        self.up1 = UpBlock(128, 64)   # 64  -> 128
        self.up2 = UpBlock(64, 32)    # 128 -> 256
        self.up3 = UpBlock(32, 16)    # 256 -> 512
        self.up4 = UpBlock(16, 8)     # 512 -> 1024
        self.head = nn.Conv2d(8, num_classes, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    x = torch.rand(1, 256, 64, 64)
    d = SemanticDecoder(num_classes=2)
    y = d(x)
    print("logits:", y.shape)          # [1,2,1024,1024]
    prob = torch.softmax(y, dim=1)
    mask = torch.argmax(prob, dim=1)
    print("mask:", mask.shape, "unique:", mask.unique())
