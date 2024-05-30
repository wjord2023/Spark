import torch
from torch import nn

from .utils import GlobalSparseAttention, SinusoidalPositionalEncoding2D


class ResidualWithAttention(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        image_size,
        patch_size,
        num_heads,
        stride=1,
    ):
        super(ResidualWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attn = GlobalSparseAttention(
            image_size, out_channels, patch_size, num_heads
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.attn(y)
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class Net(nn.Module):
    def __init__(
        self, num_classes=10, image_size=32, channel_size=3, patch_size=4, num_heads=4
    ):
        super(Net, self).__init__()
        self.num_heads = num_heads
        self.in_channels = 64

        self.pe = SinusoidalPositionalEncoding2D(image_size, image_size)
        self.attn = GlobalSparseAttention(
            image_size, channel_size + 2, patch_size, num_heads
        )

        self.conv1 = nn.Conv2d(
            channel_size + 2, self.in_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(64, 1, image_size, patch_size)
        self.layer2 = self.make_layer(128, 2, image_size // 2, patch_size, stride=2)
        self.layer3 = self.make_layer(256, 2, image_size // 4, patch_size // 2, stride=2)
        self.layer4 = self.make_layer(512, 2, image_size // 8, patch_size // 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, image_size, patch_size, stride=1):
        layers = []
        layers.append(
            ResidualWithAttention(
                self.in_channels,
                out_channels,
                image_size,
                patch_size,
                self.num_heads,
                stride,
            )
        )
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                ResidualWithAttention(
                    out_channels, out_channels, image_size, patch_size, self.num_heads
                )
            )
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pe(x)
        x = self.attn(x)
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x