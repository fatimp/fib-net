"""Run the FIB-NET inference path on a generated three-section stack."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from fibnet.inference import predict_tiled
from fibnet.model import ResUNet


def synthetic_section(offset: float, size: int = 64) -> Image.Image:
    y, x = np.mgrid[:size, :size]
    pore = (x - (31.0 + offset)) ** 2 + (y - 32.0) ** 2 < 13.0**2
    image = 170.0 + 20.0 * np.sin(y / 7.0)
    image = np.where(pore, image - 95.0, image)
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L")


def main() -> None:
    torch.manual_seed(0)
    model = ResUNet(in_channels=6, features=(8, 16, 32, 64)).eval()
    with TemporaryDirectory() as directory:
        paths = []
        for index, offset in enumerate((-1.0, 0.0, 1.0)):
            path = f"{directory}/{index}.png"
            synthetic_section(offset).save(path)
            paths.append(path)
        with torch.no_grad():
            probability = predict_tiled(
                model,
                paths[1],
                tile_size=64,
                overlap=16,
                feature_mode="stack_relief",
                device="cpu",
                previous_path=paths[0],
                next_path=paths[2],
            )
    print(
        "Synthetic inference complete: "
        f"shape={probability.shape}, dtype={probability.dtype}, "
        f"range=({probability.min():.3f}, {probability.max():.3f})"
    )


if __name__ == "__main__":
    main()
