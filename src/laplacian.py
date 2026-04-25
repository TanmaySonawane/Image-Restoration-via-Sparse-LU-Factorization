"""
laplacian.py  —  Phase 2: Discrete Laplacian system construction
=================================================================
This file converts a DamagedImage into the sparse linear system Ax = b
that the solver (solver.py) will factorise and solve.

Mathematical derivation
-----------------------
For a 2-D grayscale image, the discrete Laplace equation at interior
pixel (i, j) states that the pixel value equals the average of its four
axis-aligned neighbours:

    u[i,j] = ( u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] ) / 4

Rearranging to standard form  L·u = 0 :

    4·u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] = 0   ...(*)

We apply (*) to every MISSING pixel.  Two cases arise for each neighbour:

  Case A — neighbour is also MISSING:
      Its coefficient  (-1)  goes into matrix A at the appropriate column.

  Case B — neighbour is KNOWN (boundary condition):
      Its value is already fixed; multiply the coefficient (-1) by the
      known value and subtract from the right-hand side:
          b[row] -= (-1) * known_value  =>  b[row] += known_value

After processing all missing pixels we obtain the n×n sparse system
    A · x = b
where:
  n    = number of missing pixels
  x    = unknown pixel intensities  (what we solve for)
  A    = sparse matrix with ≤5 nonzeros per row (the Laplacian stencil)
  b    = right-hand side encoding known boundary pixel contributions

Pixel ↔ equation indexing
--------------------------
We assign each missing pixel a unique integer index 0, 1, …, n-1 by
scanning the mask row-by-row, left-to-right.  A lookup table
    pixel_to_eq[(i, j)] -> k
maps (row, col) coordinates of missing pixels to their equation index k.
This is used when filling A: a missing neighbour at (i', j') maps to
column pixel_to_eq[(i', j')].

Sparsity note
-------------
A is stored in COO (coordinate) format during assembly (cheap random
insertion) and converted to CSR before returning (required by splu).
The density of A is at most  5n / n²  =  5/n  — vanishingly sparse for
large n.  For 1 000 missing pixels the density is 0.5 %; for 10 000 it
is 0.05 %.  This is why we never build a dense n×n matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass

from damage import DamagedImage   # Phase 1 output is Phase 2 input


# ---------------------------------------------------------------------------
# Data container returned by build_system
# ---------------------------------------------------------------------------

@dataclass
class LaplacianSystem:
    """
    Holds the sparse linear system and the metadata needed to map the
    solution vector x back to a 2-D pixel array.

    Attributes
    ----------
    A          : scipy.sparse.csr_matrix, shape (n, n)
                 The discrete Laplacian operator restricted to missing pixels.
    b          : np.ndarray, shape (n,)
                 Right-hand side from known boundary pixel contributions.
    pixel_index: np.ndarray, shape (n, 2), dtype int
                 pixel_index[k] = [row, col] of the k-th unknown.
                 Use this to write solver output back into the image array.
    n          : int
                 Number of unknowns (= number of missing pixels).
    damaged    : reference to the originating DamagedImage (for Phase 5).
    """
    A          : sp.csr_matrix
    b          : np.ndarray
    pixel_index: np.ndarray
    n          : int
    damaged    : DamagedImage

    def sparsity_info(self) -> dict:
        """
        Return a dict summarising matrix sparsity for the report / terminal.
        """
        nnz   = self.A.nnz
        total = self.n ** 2
        return {
            "n_unknowns"  : self.n,
            "nnz_A"       : nnz,
            "density_A_pct": round(100 * nnz / total, 4) if total > 0 else 0.0,
            "avg_nnz_per_row": round(nnz / self.n, 2) if self.n > 0 else 0.0,
        }

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        Write the solution vector x back into a full image array.

        Parameters
        ----------
        x : np.ndarray  shape (n,)  — solver output (unknown pixel values)

        Returns
        -------
        np.ndarray  shape (H, W), float64 — the restored image, clipped to [0, 1]
        """
        restored = self.damaged.damaged.copy()
        for k, (r, c) in enumerate(self.pixel_index):
            restored[r, c] = x[k]
        return np.clip(restored, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_system(di: DamagedImage) -> LaplacianSystem:
    """
    Construct the sparse linear system  A·x = b  for the given DamagedImage.

    Algorithm (one pass through the mask):
    1.  Enumerate missing pixels, assign equation index 0 … n-1.
    2.  For each missing pixel (i, j) — equation k:
          For each of its 4 neighbours (ni, nj):
            If (ni, nj) is missing   → A[k, pixel_to_eq[(ni,nj)]] -= 1
            If (ni, nj) is known     → b[k] += damaged[ni, nj]
            If (ni, nj) is out of bounds → treat as Neumann condition
                                          (do NOT subtract; equivalent to
                                           "zero-flux" at image border).
        Always set A[k, k] = (number of in-bounds neighbours of (i,j)).

    The Neumann boundary condition at image edges means we replace
    out-of-bounds neighbours with the unknown pixel itself, so their
    contribution cancels on both sides.  In practice this slightly
    reduces the diagonal value for border pixels (from 4 to 3 or 2),
    keeping A non-singular.

    Parameters
    ----------
    di : DamagedImage from damage.py

    Returns
    -------
    LaplacianSystem
    """
    mask    = di.mask          # bool array (H, W): True = missing
    damaged = di.damaged       # float64 (H, W)
    H, W    = mask.shape

    # ------------------------------------------------------------------
    # Step 1: assign an equation index to each missing pixel
    # ------------------------------------------------------------------
    # missing_coords[k] = (row, col) of the k-th missing pixel
    missing_rows, missing_cols = np.where(mask)
    n = len(missing_rows)

    if n == 0:
        raise ValueError("Mask has no missing pixels — nothing to solve.")

    # Reverse lookup: (row, col) → equation index k
    # Using a dict is simple and correct; for very large systems a 2-D
    # integer array (eq_map) is faster but uses more memory.
    pixel_to_eq: dict[tuple[int,int], int] = {
        (int(missing_rows[k]), int(missing_cols[k])): k
        for k in range(n)
    }

    # Store pixel coordinates for later reconstruction
    pixel_index = np.stack([missing_rows, missing_cols], axis=1)  # (n, 2)

    # ------------------------------------------------------------------
    # Step 2: assemble COO data for A and dense vector b
    # ------------------------------------------------------------------
    # COO format: three lists (row indices, col indices, values)
    # We pre-allocate for the worst case of 5 entries per row.
    coo_row  = np.empty(5 * n, dtype=np.int32)
    coo_col  = np.empty(5 * n, dtype=np.int32)
    coo_data = np.empty(5 * n, dtype=np.float64)
    ptr      = 0                     # next free slot in the COO arrays

    b = np.zeros(n, dtype=np.float64)

    # The four (delta_row, delta_col) offsets for N / S / W / E neighbours
    NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for k in range(n):
        i, j = int(missing_rows[k]), int(missing_cols[k])

        # Count in-bounds neighbours to set the diagonal correctly
        # (Neumann condition: skip out-of-bounds neighbours entirely)
        diagonal = 0

        for (di_r, dj) in NEIGHBOURS:
            ni, nj = i + di_r, j + dj

            # Out-of-bounds → Neumann: skip (do not subtract this neighbour)
            if ni < 0 or ni >= H or nj < 0 or nj >= W:
                continue

            diagonal += 1   # this neighbour is in-bounds

            if mask[ni, nj]:
                # -------------------------------------------------------
                # Case A: neighbour is also MISSING
                # Coefficient -1 goes into A at column pixel_to_eq[(ni,nj)]
                # -------------------------------------------------------
                # Discrete Laplace equation (*):  -1 × u[ni,nj]  in A
                coo_row [ptr] = k
                coo_col [ptr] = pixel_to_eq[(ni, nj)]
                coo_data[ptr] = -1.0
                ptr += 1
            else:
                # -------------------------------------------------------
                # Case B: neighbour is KNOWN (boundary condition)
                # Move its contribution to the right-hand side:
                #   A·x = b  =>  b[k] += 1 × damaged[ni, nj]
                # (coefficient in (*) is -1, negated to move to RHS → +1)
                # -------------------------------------------------------
                b[k] += damaged[ni, nj]

        # Diagonal entry: coefficient of u[i,j] in equation (*) = +diagonal
        # (equals 4 for interior pixels, 3 for edge pixels, 2 for corners)
        coo_row [ptr] = k
        coo_col [ptr] = k
        coo_data[ptr] = float(diagonal)
        ptr += 1

    # Trim pre-allocated arrays to actual number of entries used
    coo_row  = coo_row [:ptr]
    coo_col  = coo_col [:ptr]
    coo_data = coo_data[:ptr]

    # ------------------------------------------------------------------
    # Step 3: build CSR matrix (required by scipy.sparse.linalg.splu)
    # ------------------------------------------------------------------
    A_coo = sp.coo_matrix(
        (coo_data, (coo_row, coo_col)),
        shape=(n, n),
        dtype=np.float64,
    )
    A_csr = A_coo.tocsr()

    return LaplacianSystem(
        A           = A_csr,
        b           = b,
        pixel_index = pixel_index,
        n           = n,
        damaged     = di,
    )


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def print_system_summary(sys: LaplacianSystem) -> None:
    """Print a concise summary of the linear system to stdout."""
    info = sys.sparsity_info()
    print("=" * 50)
    print("Laplacian system summary")
    print("=" * 50)
    print(f"  Unknowns (missing pixels) : {info['n_unknowns']:>10,}")
    print(f"  Matrix size               : {info['n_unknowns']:>6,} × {info['n_unknowns']:,}")
    print(f"  Nonzero entries in A      : {info['nnz_A']:>10,}")
    print(f"  Density of A              : {info['density_A_pct']:>10.4f} %")
    print(f"  Average nnz per row       : {info['avg_nnz_per_row']:>10.2f}")
    print("=" * 50)


def visualise_stencil_row(sys: LaplacianSystem, k: int = 0) -> None:
    """
    Print the k-th row of A as a human-readable equation.
    Useful for verifying that the Laplacian was assembled correctly.

    Example output for an interior pixel with 4 missing neighbours:
        Equation 0  [pixel (12, 15)]:
          4.0 · x[0] - 1.0 · x[1] - 1.0 · x[14] - 1.0 · x[15] - 1.0 · x[29]
          = b[0] = 0.0000
    """
    row = sys.A.getrow(k)
    r, c = sys.pixel_index[k]
    terms = []
    for col_idx, val in zip(row.indices, row.data):
        if col_idx == k:
            terms.insert(0, f"{val:+.1f}·x[{col_idx}]")
        else:
            terms.append(f"{val:+.1f}·x[{col_idx}]")
    equation_str = " ".join(terms)
    print(f"Equation {k}  [pixel ({r}, {c})]:")
    print(f"  {equation_str}")
    print(f"  = b[{k}] = {sys.b[k]:.4f}")


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly: python laplacian.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from damage import load_grayscale, damage_rectangle
    import urllib.request, tempfile, os

    # ── load a test image ──────────────────────────────────────────────
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    img = load_grayscale(tmp.name, max_side=128)   # small for quick test
    os.unlink(tmp.name)

    H, W = img.shape
    di   = damage_rectangle(img, H//4, W//4, H//4, W//4)
    sys  = build_system(di)

    print_system_summary(sys)
    print()
    visualise_stencil_row(sys, k=0)
    print()
    visualise_stencil_row(sys, k=sys.n // 2)

    # ── visualise sparsity pattern of A ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(di.damaged, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Damaged image ({di.n_missing} missing pixels)")
    axes[0].axis("off")

    axes[1].spy(sys.A, markersize=0.5, color="steelblue")
    axes[1].set_title(
        f"Sparsity pattern of A  ({sys.A.nnz} nonzeros, "
        f"{sys.sparsity_info()['density_A_pct']:.3f}% dense)"
    )
    axes[1].set_xlabel("Column (unknown index)")
    axes[1].set_ylabel("Row (equation index)")

    plt.tight_layout()
    plt.savefig("laplacian_sparsity.png", dpi=120)
    plt.show()
    print("Saved laplacian_sparsity.png")
