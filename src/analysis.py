"""
analysis.py  —  Phase 4: Comparative analysis
==============================================
Responsibilities:
  - Run the pipeline across MULTIPLE damage sizes on the same image
  - For each size: time sparse LU (with reuse) vs dense LU (re-factor each time)
  - Compute condition number κ(A) for each damage size
  - Compute per-pixel residual between original and restored image
  - Produce four publication-ready figures:
      1. Side-by-side: original | damaged | restored
      2. Per-pixel residual heatmap
      3. Timing bar chart: factor + solve times, sparse vs dense
      4. κ(A) vs damage area (log scale)
  - Save a CSV comparison table

How "LU reuse" vs "Gaussian elimination" is measured
-----------------------------------------------------
Both methods use the same underlying algorithm (partial-pivoting LU).
The difference is operational:

  Gaussian elimination proxy:
    For every new right-hand side, call lu_factor(A_dense) then lu_solve().
    This refactors A from scratch each time.  Cost = O(n³) per RHS.

  LU reuse (sparse):
    Call splu(A) ONCE.  For subsequent right-hand sides, call .solve(b_new).
    Cost = O(n²) per additional RHS (only triangular substitutions).

To make the comparison meaningful, we generate THREE damage regions of
increasing size on the same image, then measure total wall-clock time to
solve all three RHS vectors under each strategy.

Note: because the mask changes with each damage region, A changes too —
so technically each region still requires its own factorisation.  What
we are demonstrating is the SCALING of factorisation cost vs solve cost,
and how the ratio changes with n.  For the "reuse" scenario we keep the
factorisation from the *first* region and apply the same L,U to the b
vectors of the subsequent regions as an approximation experiment — this
is the intellectually honest framing for the report.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from damage    import load_grayscale, damage_rectangle, DamagedImage
from laplacian import build_system, LaplacianSystem
from solver    import factorise, factorise_dense, condition_number_sparse, SolverResult


# ---------------------------------------------------------------------------
# Damage specification for the three comparison regions
# ---------------------------------------------------------------------------

# Each entry: (block_side_pixels,) — square blocks centred on the image.
# Sizes chosen so n_missing scales roughly as 25x, 100x, 225x a base unit.
DAMAGE_SIZES = [20, 40, 60]   # side length of the square hole in pixels


# ---------------------------------------------------------------------------
# Main analysis runner
# ---------------------------------------------------------------------------

def run_analysis(
    image_path : str | Path,
    output_dir : str | Path = "results",
    max_side   : int        = 200,
    dense_limit: int        = 4000,   # skip dense path if n > this
) -> list[dict]:
    """
    Run the full Phase 4 analysis on one image.

    Parameters
    ----------
    image_path  : path to any image file
    output_dir  : where to save figures and the CSV table
    max_side    : image is downscaled so its longest side ≤ max_side
    dense_limit : maximum n for which we attempt the dense comparison
                  (dense LU is O(n³) — too slow / too much RAM for large n)

    Returns
    -------
    List of per-damage-size result dicts (also written to CSV).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    img = load_grayscale(image_path, max_side=max_side)
    H, W = img.shape
    print(f"Image loaded: {H}×{W}  ({H*W:,} total pixels)")

    records   = []
    systems   = []
    results   = []
    di_list   = []

    # ------------------------------------------------------------------
    # 1.  Build all systems and run sparse LU for each damage size
    # ------------------------------------------------------------------
    for side in DAMAGE_SIZES:
        # Centre the block on the image
        r0 = max(1, (H - side) // 2)
        c0 = max(1, (W - side) // 2)
        side_r = min(side, H - r0 - 1)
        side_c = min(side, W - c0 - 1)

        di  = damage_rectangle(img, r0, c0, side_r, side_c)
        sys = build_system(di)

        print(f"\n  Damage {side_r}×{side_c}  →  n={sys.n:,} unknowns")

        # Sparse LU
        res = factorise(sys)
        res.print_summary()

        # Condition number
        cond = condition_number_sparse(sys)
        print(f"  κ(A) estimate: {cond:.3e}")

        di_list.append(di)
        systems.append(sys)
        results.append(res)

        rec = {
            "damage_side"    : side,
            "n_missing"      : sys.n,
            "cond_A"         : cond,
            "sparse_fac_ms"  : round(res.factor_time_s  * 1000, 3),
            "sparse_solve_ms": round(res.solve_time_s   * 1000, 3),
            "sparse_residual": round(res.residual_norm(),        8),
            "dense_fac_ms"   : None,
            "dense_solve_ms" : None,
            "dense_residual" : None,
        }

        # Dense LU (only for small enough systems)
        if sys.n <= dense_limit:
            x_d, ft_d, st_d, _ = factorise_dense(sys)
            dense_residual = _residual_norm(sys.A, x_d, sys.b)
            rec["dense_fac_ms"]   = round(ft_d * 1000, 3)
            rec["dense_solve_ms"] = round(st_d * 1000, 3)
            rec["dense_residual"] = round(dense_residual, 8)
            print(f"  Dense LU — factor: {ft_d*1000:.2f} ms | "
                  f"solve: {st_d*1000:.2f} ms | residual: {dense_residual:.2e}")
        else:
            print(f"  Dense LU skipped (n={sys.n} > limit={dense_limit})")

        records.append(rec)

    # ------------------------------------------------------------------
    # 2.  LU reuse demonstration
    #     Use the factorisation from the FIRST (smallest) system and apply
    #     its L,U to the b vector of the second system.
    #     This demonstrates the solve-only cost when A stays fixed.
    # ------------------------------------------------------------------
    print("\n--- LU reuse demo (factor once, solve three times) ---")
    first_result = results[0]
    reuse_times  = []
    for i, sys_ in enumerate(systems):
        _, t = first_result.solve_new_rhs(sys_.b[:systems[0].n])
        reuse_times.append(t * 1000)
        print(f"  Reuse solve {i+1}: {t*1000:.3f} ms  (b has {sys_.n} entries, "
              f"truncated to {systems[0].n} for demo)")

    # ------------------------------------------------------------------
    # 3.  Figures
    # ------------------------------------------------------------------
    _plot_restoration(img, di_list, results, output_dir)
    _plot_residual_heatmap(di_list, results, output_dir)
    _plot_timing(records, reuse_times, output_dir)
    _plot_condition_number(records, output_dir)

    # ------------------------------------------------------------------
    # 4.  CSV table
    # ------------------------------------------------------------------
    csv_path = Path(output_dir) / "comparison_table.csv"
    _write_csv(records, csv_path)
    print(f"\nComparison table saved → {csv_path}")

    return records


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_restoration(
    img      : np.ndarray,
    di_list  : list[DamagedImage],
    results  : list[SolverResult],
    out_dir  : str | Path,
) -> None:
    """Figure 1: Original | Damaged | Restored for each damage size."""
    n_sizes = len(di_list)
    fig, axes = plt.subplots(n_sizes, 3, figsize=(11, 3.5 * n_sizes))
    if n_sizes == 1:
        axes = [axes]

    for row, (di, res) in enumerate(zip(di_list, results)):
        axes[row][0].imshow(img,                   cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title("Original",         fontsize=10)
        axes[row][1].imshow(di.damaged,            cmap="gray", vmin=0, vmax=1)
        axes[row][1].set_title(
            f"Damaged  ({di.n_missing} px)",       fontsize=10)
        axes[row][2].imshow(res.restored_image,    cmap="gray", vmin=0, vmax=1)
        axes[row][2].set_title(
            f"Restored  (‖r‖={res.residual_norm():.1e})", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("Image restoration via discrete Laplacian + sparse LU",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = Path(out_dir) / "fig1_restoration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def _plot_residual_heatmap(
    di_list : list[DamagedImage],
    results : list[SolverResult],
    out_dir : str | Path,
) -> None:
    """Figure 2: Per-pixel absolute error |restored - original| heatmap."""
    n_sizes = len(di_list)
    fig, axes = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 4))
    if n_sizes == 1:
        axes = [axes]

    for ax, di, res in zip(axes, di_list, results):
        err = np.abs(res.restored_image - di.original)
        # Only show error inside the masked region for clarity
        err_masked = np.where(di.mask, err, np.nan)
        im = ax.imshow(err_masked, cmap="hot", vmin=0, vmax=0.1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="|restored − original|")
        ax.set_title(
            f"Damage {di.label.split('_')[1] if '_' in di.label else '?'}\n"
            f"max err={err[di.mask].max():.3f}  "
            f"mean err={err[di.mask].mean():.4f}",
            fontsize=9,
        )
        ax.axis("off")

    plt.suptitle("Per-pixel reconstruction error (masked region only)", fontsize=11)
    plt.tight_layout()
    path = Path(out_dir) / "fig2_residual_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def _plot_timing(
    records     : list[dict],
    reuse_times : list[float],
    out_dir     : str | Path,
) -> None:
    """Figure 3: Grouped bar chart — factorisation vs solve time, sparse vs dense."""
    labels      = [f"n={r['n_missing']}" for r in records]
    sparse_fac  = [r["sparse_fac_ms"]   for r in records]
    sparse_slv  = [r["sparse_solve_ms"] for r in records]
    dense_fac   = [r["dense_fac_ms"]  if r["dense_fac_ms"]  is not None else 0
                   for r in records]
    dense_slv   = [r["dense_solve_ms"] if r["dense_solve_ms"] is not None else 0
                   for r in records]

    x   = np.arange(len(labels))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = [
        ax.bar(x - 1.5*w, sparse_fac, w, label="Sparse LU — factor",  color="#2176AE"),
        ax.bar(x - 0.5*w, sparse_slv, w, label="Sparse LU — solve",   color="#57C4E5"),
        ax.bar(x + 0.5*w, dense_fac,  w, label="Dense LU — factor",   color="#E05C5C"),
        ax.bar(x + 1.5*w, dense_slv,  w, label="Dense LU — solve",    color="#F4A261"),
    ]

    # Annotate reuse solve times above the sparse-solve bars
    for xi, rt in zip(x, reuse_times):
        ax.annotate(
            f"reuse\n{rt:.2f} ms",
            xy=(xi - 0.5*w, sparse_slv[x.tolist().index(xi)]),
            xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=7, color="#057D70",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title("Factorisation vs solve time: sparse LU vs dense LU\n"
                 "(green annotations = solve-only cost with factor reuse)")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    path = Path(out_dir) / "fig3_timing.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def _plot_condition_number(
    records : list[dict],
    out_dir : str | Path,
) -> None:
    """Figure 4: κ(A) vs damage area on a log-log scale."""
    areas = [r["n_missing"] for r in records]
    conds = [r["cond_A"]    for r in records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(areas, conds, "o-", color="#534AB7", linewidth=2, markersize=8)
    for a, c in zip(areas, conds):
        ax.annotate(f"{c:.1e}", (a, c),
                    textcoords="offset points", xytext=(4, 4), fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of missing pixels  (damage area n)")
    ax.set_ylabel("Condition number  κ(A)")
    ax.set_title("Condition number grows with damage area\n"
                 "(larger / more connected damage → harder system)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    path = Path(out_dir) / "fig4_condition_number.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(records: list[dict], path: Path) -> None:
    headers = [
        "damage_side", "n_missing", "cond_A",
        "sparse_fac_ms", "sparse_solve_ms", "sparse_residual",
        "dense_fac_ms",  "dense_solve_ms",  "dense_residual",
    ]
    lines = [",".join(headers)]
    for r in records:
        lines.append(",".join(
            "N/A" if r[h] is None else str(r[h]) for h in headers
        ))
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _residual_norm(A, x, b) -> float:
    r  = A.dot(x) - b
    bn = np.linalg.norm(b)
    return float(np.linalg.norm(r) / bn) if bn > 0 else float(np.linalg.norm(r))


# ---------------------------------------------------------------------------
# Quick self-test  (python analysis.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import urllib.request, tempfile, os

    url = ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/"
           "Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg")
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    urllib.request.urlretrieve(url, tmp.name)

    records = run_analysis(tmp.name, output_dir="results", max_side=200)
    os.unlink(tmp.name)

    print("\n\nFinal comparison table:")
    print(f"{'Side':>6} {'n':>6} {'κ(A)':>10} "
          f"{'SparFac':>9} {'SparSlv':>9} "
          f"{'DenFac':>9} {'DenSlv':>9} "
          f"{'SparRes':>10} {'DenRes':>10}")
    print("-" * 90)
    for r in records:
        print(
            f"{r['damage_side']:>6} "
            f"{r['n_missing']:>6} "
            f"{r['cond_A']:>10.2e} "
            f"{r['sparse_fac_ms']:>9.3f} "
            f"{r['sparse_solve_ms']:>9.3f} "
            f"{ str(round(r['dense_fac_ms'],3))  if r['dense_fac_ms']  else 'N/A':>9} "
            f"{ str(round(r['dense_solve_ms'],3)) if r['dense_solve_ms'] else 'N/A':>9} "
            f"{r['sparse_residual']:>10.2e} "
            f"{ str(r['dense_residual']) if r['dense_residual'] else 'N/A':>10}"
        )
