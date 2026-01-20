import numpy as np
from pathlib import Path

from utils.document_loader import load_document_as_images
from utils.config import IMAGE_SIZE


def test_load_image_document():
    test_image_path = Path("data/images")

    image_files = (
        list(test_image_path.glob("*.png")) +
        list(test_image_path.glob("*.jpg")) +
        list(test_image_path.glob("*.jpeg"))
    )

    assert len(image_files) > 0, "No test images found"

    images = load_document_as_images(str(image_files[0]))

    assert isinstance(images, list)
    assert len(images) == 1

    img = images[0]

    assert isinstance(img, np.ndarray)
    assert img.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0])
    assert img.min() >= 0.0
    assert img.max() <= 1.0
