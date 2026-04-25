"""
pipeline.py  —  Phases 1-5 orchestrator
========================================
This is the single entry point for running the entire project locally.
It calls damage.py → laplacian.py → solver.py → analysis.py in order,
saves all results, and prints a clean progress log to the terminal.

Run from the project root:
    python src/pipeline.py
    python src/pipeline.py --image assets/sample_image.png
    python src/pipeline.py --image assets/sample_image.png --max-side 300
    python src/pipeline.py --image assets/sample_image.png --sizes 15 35 55

This file does NOT contain any solver logic.  It is a thin orchestration
layer that wires the four modules together and handles CLI arguments,
output paths, and a final summary print.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make imports work whether pipeline.py is run from project root or src/
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from damage    import load_grayscale, damage_rectangle, damage_diagonal_scratch
from laplacian import build_system, print_system_summary
from solver    import factorise, condition_number_sparse
from analysis  import run_analysis, DAMAGE_SIZES


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_IMAGE   = _here.parent / "assets" / "sample_image.png"
DEFAULT_OUT_DIR = _here.parent / "results"
DEMO_IMAGE_URL  = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/"
    "Camponotus_flavomarginatus_ant.jpg/"
    "320px-Camponotus_flavomarginatus_ant.jpg"
)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Image restoration via discrete Laplacian + sparse LU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image", type=Path, default=None,
        help="Path to input image.  If omitted, a demo image is downloaded.",
    )
    p.add_argument(
        "--max-side", type=int, default=256,
        help="Downscale image so its longest side ≤ this value.",
    )
    p.add_argument(
        "--sizes", type=int, nargs="+", default=None,
        help=(
            "Side lengths (pixels) of the square damage blocks used in the "
            "comparison analysis.  Overrides the default list in analysis.py."
        ),
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Directory where all figures, tables, and restored images are saved.",
    )
    p.add_argument(
        "--dense-limit", type=int, default=4000,
        help=(
            "Maximum number of unknowns n for which the dense LU baseline "
            "is attempted.  Dense LU is O(n³); skip for large n."
        ),
    )
    p.add_argument(
        "--scratch", action="store_true",
        help="Also run a diagonal scratch demo in addition to the block analysis.",
    )
    return p


# ---------------------------------------------------------------------------
# Single-image demo (one damage type, one solve, one figure)
# ---------------------------------------------------------------------------

def run_demo(
    image        : np.ndarray,
    out_dir      : Path,
    damage_type  : str = "rect",
    damage_kwargs: dict | None = None,
) -> None:
    """
    Demonstrate the full pipeline on a single damage region.
    Produces one 3-panel figure (original | damaged | restored).

    Parameters
    ----------
    image         : float64 greyscale array (H, W)
    out_dir       : where to save the figure
    damage_type   : "rect" or "scratch"
    damage_kwargs : extra kwargs forwarded to the damage function
    """
    H, W = image.shape
    kw   = damage_kwargs or {}

    _banner(f"Demo — {damage_type} damage")

    # Phase 1 — damage
    if damage_type == "rect":
        defaults = dict(row_start=H//4, col_start=W//4,
                        height=H//4,    width=W//4)
        defaults.update(kw)
        di = damage_rectangle(image, **defaults)
    else:
        defaults = dict(thickness=4)
        defaults.update(kw)
        di = damage_diagonal_scratch(image, **defaults)

    print(f"  Damage applied  : {di.label}")
    print(f"  Missing pixels  : {di.n_missing:,}  "
          f"({100*di.n_missing/(H*W):.1f}% of image)")

    # Phase 2 — build system
    _banner("Phase 2 — Laplacian system")
    sys_ = build_system(di)
    print_system_summary(sys_)

    # Phase 3 — solve
    _banner("Phase 3 — Sparse LU solve")
    result = factorise(sys_)
    result.print_summary()

    cond = condition_number_sparse(sys_)
    print(f"  κ(A): {cond:.3e}")

    # Phase 5 — save figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(image,                  cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original",          fontsize=11)
    axes[1].imshow(di.damaged,             cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"Damaged  ({di.n_missing:,} px missing)", fontsize=11)
    axes[2].imshow(result.restored_image,  cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"Restored  ‖r‖ = {result.residual_norm():.1e}", fontsize=11)
    for ax in axes:
        ax.axis("off")
    plt.suptitle(
        f"Image restoration — {damage_type} damage  |  "
        f"n={sys_.n:,} unknowns  |  κ(A)={cond:.1e}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    out = out_dir / f"demo_{damage_type}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {out}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # ------------------------------------------------------------------
    # Resolve / download image
    # ------------------------------------------------------------------
    image_path = args.image
    if image_path is None or not image_path.exists():
        if image_path is not None:
            print(f"[warn] Image not found: {image_path}")
        image_path = _download_demo_image()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory : {out_dir.resolve()}")

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------
    _banner("Loading image")
    image = load_grayscale(image_path, max_side=args.max_side)
    H, W  = image.shape
    print(f"  Path    : {image_path}")
    print(f"  Size    : {H} × {W}  ({H*W:,} pixels)")

    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # Single-image demo (rectangle)
    # ------------------------------------------------------------------
    run_demo(image, out_dir, damage_type="rect")

    # ------------------------------------------------------------------
    # Optional scratch demo
    # ------------------------------------------------------------------
    if args.scratch:
        run_demo(image, out_dir, damage_type="scratch")

    # ------------------------------------------------------------------
    # Full comparative analysis (Phases 1-4)
    # ------------------------------------------------------------------
    _banner("Phase 4 — Comparative analysis")

    # Allow CLI override of damage sizes
    if args.sizes:
        import analysis as _ana
        _ana.DAMAGE_SIZES = args.sizes
        print(f"  Using custom damage sizes: {args.sizes}")
    else:
        print(f"  Using default damage sizes: {DAMAGE_SIZES}")

    records = run_analysis(
        image_path  = image_path,
        output_dir  = out_dir,
        max_side    = args.max_side,
        dense_limit = args.dense_limit,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_total
    _banner(f"Pipeline complete  ({elapsed:.2f} s total)")
    print(f"  Results saved to : {out_dir.resolve()}")
    print()
    print(f"  {'Damage':>8}  {'n':>6}  {'κ(A)':>10}  "
          f"{'SparFac ms':>12}  {'SparSlv ms':>12}  {'Residual':>10}")
    print("  " + "-" * 68)
    for r in records:
        print(
            f"  {r['damage_side']:>8}  "
            f"{r['n_missing']:>6,}  "
            f"{r['cond_A']:>10.2e}  "
            f"{r['sparse_fac_ms']:>12.3f}  "
            f"{r['sparse_solve_ms']:>12.3f}  "
            f"{r['sparse_residual']:>10.2e}"
        )
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_demo_image() -> Path:
    """Download a public-domain test image to assets/ if not present."""
    import urllib.request

    assets = _here.parent / "assets"
    assets.mkdir(exist_ok=True)
    dest = assets / "sample_image.png"

    if dest.exists():
        print(f"  Using cached demo image: {dest}")
        return dest

    print(f"  Downloading demo image from Wikipedia …")
    try:
        urllib.request.urlretrieve(DEMO_IMAGE_URL, dest)
        print(f"  Saved → {dest}")
    except Exception as exc:
        print(f"  Download failed ({exc}). "
              "Place any image at assets/sample_image.png and retry.")
        sys.exit(1)
    return dest


def _banner(title: str) -> None:
    print()
    print("─" * 54)
    print(f"  {title}")
    print("─" * 54)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
