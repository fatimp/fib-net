from pathlib import Path

import numpy as np
import torch
from PIL import Image

import fibnet
from fibnet.dataset import mask_to_array
from fibnet.ensemble import arithmetic_probability_mean
from fibnet.image_features import image_stack_to_feature_array
from fibnet.inference import predict_tiled
from fibnet.metrics import dice_coefficient, iou_score
from fibnet.model import ResUNet
from fibnet.probability_io import load_probability_map, save_probability_map


def gray_image(value: int, shape: tuple[int, int] = (32, 40)) -> Image.Image:
    return Image.fromarray(np.full(shape, value, dtype=np.uint8), mode="L")


def test_public_package_import() -> None:
    assert fibnet.__version__ == "0.1.0"


def test_stack_relief_builds_six_float32_channels() -> None:
    features = image_stack_to_feature_array(
        gray_image(32), gray_image(96), gray_image(160)
    )

    assert features.shape == (6, 32, 40)
    assert features.dtype == np.float32
    assert not np.array_equal(features[0], features[1])
    assert not np.array_equal(features[1], features[2])


def test_resunet_six_channel_forward_shape() -> None:
    model = ResUNet(in_channels=6, features=(8, 16, 32, 64))

    output = model(torch.zeros((2, 6, 32, 48), dtype=torch.float32))

    assert output.shape == (2, 1, 32, 48)


def test_tiled_inference_on_synthetic_stack(tmp_path: Path) -> None:
    paths = []
    for index, value in enumerate((32, 96, 160)):
        path = tmp_path / f"{index}.png"
        gray_image(value, shape=(35, 43)).save(path)
        paths.append(path)
    model = torch.nn.Conv2d(6, 1, kernel_size=1)

    with torch.no_grad():
        probability = predict_tiled(
            model,
            paths[1],
            tile_size=32,
            overlap=8,
            feature_mode="stack_relief",
            device="cpu",
            previous_path=paths[0],
            next_path=paths[2],
        )

    assert probability.shape == (35, 43)
    assert probability.dtype == np.float32
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_probability_map_round_trip_is_float32(tmp_path: Path) -> None:
    expected = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5)
    path = save_probability_map(tmp_path / "probability.npy", expected)

    actual = load_probability_map(path)

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_arithmetic_probability_ensemble() -> None:
    first = np.full((3, 4), 0.2, dtype=np.float32)
    second = np.full((3, 4), 0.8, dtype=np.float32)

    result = arithmetic_probability_mean([first, second])

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.5)


def test_overlap_metrics_and_black_pore_mask_convention() -> None:
    manual = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    target = torch.from_numpy(mask_to_array(Image.fromarray(manual), 0))[None, None]
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])

    assert dice_coefficient(logits, target) > 0.999
    assert iou_score(logits, target) > 0.999
