import numpy as np

from fibnet.interactive import PORE, SOLID, UNLABELED
from scripts.evaluate_rf_overlay import extract_overlay_scribbles


def test_overlay_colors_are_mapped_to_semantic_labels() -> None:
    overlay = np.array(
        [
            [
                [34, 177, 76, 255],
                [237, 28, 36, 255],
                [80, 80, 80, 255],
            ],
            [
                [57, 85, 65, 255],
                [160, 130, 132, 255],
                [20, 40, 90, 255],
            ],
        ],
        dtype=np.uint8,
    )

    scribbles = extract_overlay_scribbles(overlay)

    np.testing.assert_array_equal(
        scribbles,
        [[PORE, SOLID, UNLABELED], [PORE, SOLID, UNLABELED]],
    )


def test_alpha_channel_does_not_change_semantic_labels() -> None:
    overlay = np.array([[[0, 255, 0, 0], [255, 0, 0, 1]]], dtype=np.uint8)

    scribbles = extract_overlay_scribbles(overlay)

    np.testing.assert_array_equal(scribbles, [[PORE, SOLID]])
