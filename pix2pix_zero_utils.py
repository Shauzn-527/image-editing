from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def make_row_grid(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("images must be non-empty")

    heights = [im.height for im in images]
    if len(set(heights)) != 1:
        target_h = heights[0]
        resized = []
        for im in images:
            target_w = int(round(im.width * (target_h / im.height)))
            resized.append(im.resize((target_w, target_h), resample=Image.BICUBIC))
        images = resized

    widths = [im.width for im in images]
    grid = Image.new("RGB", (sum(widths), images[0].height))
    x = 0
    for im in images:
        grid.paste(im, (x, 0))
        x += im.width
    return grid


def to_uint8_np(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    return arr.astype(np.uint8, copy=False)
