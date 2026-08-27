"""Content hashing helpers used for provenance and duplicate review."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def _dct_basis(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float32)
    frequencies = positions[:, None]
    basis = np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * size))
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis * np.sqrt(2.0 / size)


_DCT_32 = _dct_basis(32)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: str | Path, size: int = 8) -> str:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def perceptual_hash(path: str | Path, image_size: int = 32, low_frequency_size: int = 8) -> str:
    """DCT perceptual hash used only to flag candidates for human/data review."""
    if image_size != 32:
        basis = _dct_basis(image_size)
    else:
        basis = _DCT_32
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((image_size, image_size)), dtype=np.float32)
    transformed = basis @ pixels @ basis.T
    low = transformed[:low_frequency_size, :low_frequency_size]
    threshold = float(np.median(low.ravel()[1:]))
    bits = low > threshold
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{low_frequency_size * low_frequency_size // 4}x}"
