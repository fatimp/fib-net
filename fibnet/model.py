from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        if len(features) != 4:
            raise ValueError(
                "This UNet implementation expects exactly four feature levels."
            )

        self.stem = DoubleConv(in_channels, features[0])
        self.down_blocks = nn.ModuleList(
            DownBlock(features[i], features[i + 1]) for i in range(len(features) - 1)
        )
        self.bottleneck = DownBlock(features[-1], features[-1] * 2)
        decoder_in_channels = [
            features[-1] * 2,
            features[-1],
            features[-2],
            features[-3],
        ]
        decoder_skip_channels = [features[-1], features[-2], features[-3], features[-4]]
        decoder_out_channels = [features[-1], features[-2], features[-3], features[-4]]

        self.up_blocks = nn.ModuleList(
            UpBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(
                decoder_in_channels,
                decoder_skip_channels,
                decoder_out_channels,
                strict=False,
            )
        )
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []

        x = self.stem(x)
        skips.append(x)

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)
        skips = list(reversed(skips))

        for up, skip in zip(self.up_blocks, skips, strict=False):
            x = up(x, skip)

        return self.head(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(x) + self.skip(x))


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        if len(features) != 4:
            raise ValueError(
                "This ResUNet implementation expects exactly four feature levels."
            )

        self.stem = ResidualConvBlock(in_channels, features[0])
        self.down_blocks = nn.ModuleList(
            ResidualDownBlock(features[i], features[i + 1])
            for i in range(len(features) - 1)
        )
        self.bottleneck = ResidualDownBlock(features[-1], features[-1] * 2)
        decoder_in_channels = [
            features[-1] * 2,
            features[-1],
            features[-2],
            features[-3],
        ]
        decoder_skip_channels = [features[-1], features[-2], features[-3], features[-4]]
        decoder_out_channels = [features[-1], features[-2], features[-3], features[-4]]
        self.up_blocks = nn.ModuleList(
            ResidualUpBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(
                decoder_in_channels,
                decoder_skip_channels,
                decoder_out_channels,
                strict=False,
            )
        )
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        x = self.stem(x)
        skips.append(x)
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up, skip in zip(self.up_blocks, reversed(skips), strict=False):
            x = up(x, skip)
        return self.head(x)


def build_model(model_arch: str, in_channels: int, out_channels: int = 1) -> nn.Module:
    if model_arch == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels)
    if model_arch == "resunet":
        return ResUNet(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"Unsupported model_arch: {model_arch!r}")
