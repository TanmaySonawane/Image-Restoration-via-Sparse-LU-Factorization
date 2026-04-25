"""
damage.py  —  Phase 1: Image loading and damage simulation
================================================================
Responsibilities:
  - Load a grayscale image from disk and normalise to float [0, 1]
  - Introduce synthetic damage (rectangular blocks and diagonal scratches)
  - Build a binary mask array (1 = missing pixel, 0 = known pixel)
  - Return everything the Laplacian builder (laplacian.py) needs

No solvers or matrix operations live here.
All pixel values are kept as float64 throughout so that the solver
in later phases never has to deal with integer rounding.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data container returned by every damage function
# ---------------------------------------------------------------------------

@dataclass
class DamagedImage:
    """
    Bundles everything downstream phases need.

    Attributes
    ----------
    original : np.ndarray  shape (H, W), float64 in [0, 1]
        The clean image before any damage was applied.
        Used only for computing residuals in Phase 5.
    damaged  : np.ndarray  shape (H, W), float64 in [0, 1]
        Original with missing pixels zeroed out.
        The zeros are placeholders — they carry no information.
    mask     : np.ndarray  shape (H, W), dtype bool
        True  → pixel is MISSING (unknown, must be solved for)
        False → pixel is KNOWN   (boundary condition)
    label    : str
        Human-readable description of the damage, e.g. "block_40x40".
    """
    original : np.ndarray
    damaged  : np.ndarray
    mask     : np.ndarray
    label    : str = "unlabelled"

    @property
    def height(self) -> int:
        return self.original.shape[0]

    @property
    def width(self) -> int:
        return self.original.shape[1]

    @property
    def n_missing(self) -> int:
        """Number of pixels that need to be recovered."""
        return int(self.mask.sum())


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_grayscale(path: str | Path, max_side: int = 256) -> np.ndarray:
    """
    Load an image from disk, convert to grayscale, and return as float64
    array with values in [0, 1].

    Parameters
    ----------
    path     : path to any image file Pillow can open (PNG, JPG, BMP, …)
    max_side : if either dimension exceeds this, the image is downscaled
               proportionally.  Keeps the linear system tractable on a
               laptop — a 512×512 image with 30 % damage gives ~78 000
               unknowns, which is already a substantial sparse system.

    Returns
    -------
    np.ndarray  shape (H, W), dtype float64, values in [0.0, 1.0]
    """
    img = Image.open(path).convert("L")          # "L" = 8-bit greyscale

    # Downscale if needed, preserving aspect ratio
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


# ---------------------------------------------------------------------------
# Damage generators
# ---------------------------------------------------------------------------

def damage_rectangle(
    image    : np.ndarray,
    row_start: int,
    col_start: int,
    height   : int,
    width    : int,
) -> DamagedImage:
    """
    Zero out a rectangular block of pixels and mark them as missing.

    This is the most common damage pattern in the literature.
    A single rectangle with a 1-pixel-wide intact border on every side
    guarantees that every missing pixel has at least one known neighbour,
    which keeps the Laplacian system non-singular.

    Parameters
    ----------
    image     : float64 array (H, W) — the clean image
    row_start : top edge of the damaged rectangle (0-indexed)
    col_start : left edge of the damaged rectangle (0-indexed)
    height    : number of rows to damage
    width     : number of columns to damage

    Returns
    -------
    DamagedImage
    """
    H, W = image.shape
    # Clamp to valid pixel range
    r0 = max(0, row_start)
    c0 = max(0, col_start)
    r1 = min(H, row_start + height)
    c1 = min(W, col_start + width)

    mask    = np.zeros((H, W), dtype=bool)
    damaged = image.copy()

    mask   [r0:r1, c0:c1] = True
    damaged[r0:r1, c0:c1] = 0.0

    label = f"rect_{height}x{width}_at_({r0},{c0})"
    return DamagedImage(original=image, damaged=damaged, mask=mask, label=label)


def damage_diagonal_scratch(
    image      : np.ndarray,
    thickness  : int = 3,
    row_start  : int = 0,
    col_start  : int = 0,
    row_end    : int | None = None,
    col_end    : int | None = None,
) -> DamagedImage:
    """
    Draw a thick diagonal scratch across the image (or a portion of it)
    using Bresenham's line algorithm, then mark those pixels as missing.

    Parameters
    ----------
    image     : float64 array (H, W)
    thickness : width of the scratch in pixels (applied symmetrically)
    row_start, col_start : start point (defaults to top-left corner)
    row_end,   col_end   : end   point (defaults to bottom-right corner)

    Returns
    -------
    DamagedImage
    """
    H, W = image.shape
    r0, c0 = row_start, col_start
    r1 = H - 1 if row_end  is None else row_end
    c1 = W - 1 if col_end  is None else col_end

    mask    = np.zeros((H, W), dtype=bool)
    damaged = image.copy()

    # Bresenham's line — integer step rasterisation
    line_pixels = _bresenham(r0, c0, r1, c1)

    half = thickness // 2
    for (r, c) in line_pixels:
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    mask   [rr, cc] = True
                    damaged[rr, cc] = 0.0

    label = f"scratch_t{thickness}_({r0},{c0})->({r1},{c1})"
    return DamagedImage(original=image, damaged=damaged, mask=mask, label=label)


def damage_multi(
    image  : np.ndarray,
    specs  : list[dict],
) -> list[DamagedImage]:
    """
    Apply several independent damage regions to the same clean image.

    Each entry in `specs` is a dict with a "type" key and keyword arguments
    for the corresponding damage function.  The clean image is reused for
    every spec so that damage regions do not compound.

    Example
    -------
    specs = [
        {"type": "rect",     "row_start": 30,  "col_start": 30,
                              "height": 20,     "width": 20},
        {"type": "rect",     "row_start": 60,  "col_start": 60,
                              "height": 40,     "width": 40},
        {"type": "scratch",  "thickness": 3},
    ]
    damaged_list = damage_multi(image, specs)

    The three DamagedImage objects can then be passed individually to
    laplacian.py and solver.py to demonstrate LU factor reuse in Phase 4
    (the matrix A changes with each mask, so a new factorisation is needed
    per mask — but comparing factorisation time vs. solve time is the point).

    Returns
    -------
    list[DamagedImage]  — one per spec, all referencing the same original
    """
    results = []
    for spec in specs:
        kind = spec.pop("type")
        if kind == "rect":
            results.append(damage_rectangle(image, **spec))
        elif kind == "scratch":
            results.append(damage_diagonal_scratch(image, **spec))
        else:
            raise ValueError(f"Unknown damage type: '{kind}'. Use 'rect' or 'scratch'.")
        spec["type"] = kind          # restore so caller's dict is unchanged
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """
    Bresenham's line algorithm.
    Returns list of (row, col) integer pixel coordinates along the line
    from (r0, c0) to (r1, c1), inclusive of both endpoints.
    """
    pixels = []
    dr = abs(r1 - r0);  sr = 1 if r0 < r1 else -1
    dc = abs(c1 - c0);  sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        pixels.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc;  r += sr
        if e2 <  dr:
            err += dr;  c += sc
    return pixels


# ---------------------------------------------------------------------------
# Quick visual check (run this file directly to verify everything works)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import urllib.request, tempfile, os

    # Download a public-domain test image (Lena/cameraman substitute)
    url  = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
    tmp  = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    urllib.request.urlretrieve(url, tmp.name)

    img = load_grayscale(tmp.name, max_side=256)
    os.unlink(tmp.name)

    H, W = img.shape
    di_rect    = damage_rectangle(img, H//4, W//4, H//4, W//4)
    di_scratch = damage_diagonal_scratch(img, thickness=4)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img,              cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[1].imshow(di_rect.damaged,  cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Rectangle damage\n({di_rect.n_missing} missing pixels)")
    axes[2].imshow(di_scratch.damaged, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Diagonal scratch\n({di_scratch.n_missing} missing pixels)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("damage_check.png", dpi=120)
    plt.show()
    print("Saved damage_check.png")
    print(f"Image size      : {H} x {W}")
    print(f"Rectangle missing: {di_rect.n_missing} pixels")
    print(f"Scratch missing  : {di_scratch.n_missing} pixels")
