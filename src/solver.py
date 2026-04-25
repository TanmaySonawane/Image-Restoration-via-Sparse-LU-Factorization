"""
solver.py  —  Phase 3: LU factorisation and sparse solve
=========================================================
Responsibilities:
  - Accept the sparse system A·x = b from laplacian.py
  - Factorise A = P·L·U using scipy's sparse LU (splu)
  - Solve via forward / back substitution
  - Measure and report fill-in: how many new nonzeros appear in L and U
    compared to A (directly addresses the professor's sparsity remark)
  - Expose a reusable SolverResult object so analysis.py can call
    .solve(b_new) on an existing factorisation without re-factoring

Design note on "Gaussian elimination" vs LU reuse
--------------------------------------------------
In this project, "Gaussian elimination" means computing a fresh
factorisation for every new right-hand side.  "LU reuse" means calling
splu once and then calling .solve(b) for every subsequent right-hand
side.  The factorisation itself is identical in both cases (splu always
does partial-pivoting LU); only the timing differs.  analysis.py exploits
this to produce the comparison table the professor asked for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from laplacian import LaplacianSystem


# ---------------------------------------------------------------------------
# Data container returned by factorise()
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """
    Wraps a completed LU factorisation and its first solution.

    Attributes
    ----------
    x               : solution vector for the original b
    restored_image  : full-size float64 image with missing pixels filled in
    factor_time_s   : wall-clock seconds spent factorising A
    solve_time_s    : wall-clock seconds spent on the triangular solves
    fill_info       : dict with fill-in statistics (see _fill_info)
    _lu             : internal SuperLU object — call .solve(b_new) to reuse
    _system         : reference to the LaplacianSystem (for reconstruct)
    """
    x              : np.ndarray
    restored_image : np.ndarray
    factor_time_s  : float
    solve_time_s   : float
    fill_info      : dict
    _lu            : spla.SuperLU   = field(repr=False)
    _system        : LaplacianSystem = field(repr=False)

    # ------------------------------------------------------------------
    # Public API for LU reuse (used by analysis.py Phase 4)
    # ------------------------------------------------------------------

    def solve_new_rhs(self, b_new: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Solve A·x = b_new using the ALREADY COMPUTED L and U factors.
        No refactorisation — only two triangular substitutions.

        Returns
        -------
        (x_new, solve_time_s)
        """
        t0    = time.perf_counter()
        x_new = self._lu.solve(b_new)
        return x_new, time.perf_counter() - t0

    def residual_norm(self, x: np.ndarray | None = None) -> float:
        """
        Compute  ||A·x - b||_2 / ||b||_2  (relative residual).
        Uses self.x if x is not supplied.
        """
        xv = self.x if x is None else x
        A  = self._system.A
        b  = self._system.b
        r  = A.dot(xv) - b
        bn = np.linalg.norm(b)
        return float(np.linalg.norm(r) / bn) if bn > 0 else float(np.linalg.norm(r))

    def print_summary(self) -> None:
        fi = self.fill_info
        print("=" * 52)
        print("Solver summary")
        print("=" * 52)
        print(f"  Unknowns                  : {fi['n']:>10,}")
        print(f"  nnz in A                  : {fi['nnz_A']:>10,}")
        print(f"  nnz in L                  : {fi['nnz_L']:>10,}")
        print(f"  nnz in U                  : {fi['nnz_U']:>10,}")
        print(f"  Fill-in factor (L+U)/A    : {fi['fillin_factor']:>10.2f}x")
        print(f"  Fill-in pct above A       : {fi['fillin_pct']:>9.1f} %")
        print(f"  Factorisation time        : {self.factor_time_s*1000:>9.2f} ms")
        print(f"  Solve time                : {self.solve_time_s*1000:>9.2f} ms")
        print(f"  Relative residual ||Ax-b||: {self.residual_norm():.2e}")
        print("=" * 52)


# ---------------------------------------------------------------------------
# Core factorisation function
# ---------------------------------------------------------------------------

def factorise(system: LaplacianSystem) -> SolverResult:
    """
    Factorise system.A using sparse LU (partial pivoting + AMD reordering)
    and immediately solve for system.b.

    Parameters
    ----------
    system : LaplacianSystem from laplacian.build_system()

    Returns
    -------
    SolverResult  (holds the LU object for reuse, solution, and fill-in stats)
    """
    A = system.A
    b = system.b

    # ------------------------------------------------------------------
    # Factorisation:  A = P · L · U
    # splu uses:
    #   - Approximate Minimum Degree (AMD) column reordering to reduce fill-in
    #   - Partial pivoting for numerical stability
    #   - Compressed Sparse Column internally (CSC); we pass CSR and scipy
    #     converts automatically
    # ------------------------------------------------------------------
    # splu works internally in CSC format; convert explicitly to silence
    # SparseEfficiencyWarning (no algorithmic difference, just avoids the
    # implicit copy warning that appears when CSR is passed in).
    t_fac_start = time.perf_counter()
    lu          = spla.splu(A.tocsc())
    factor_time = time.perf_counter() - t_fac_start

    # ------------------------------------------------------------------
    # Solve:  L·c = P·b  (forward substitution)
    #         U·x = c    (back substitution)
    # splu.solve() does both steps in one call.
    # ------------------------------------------------------------------
    t_solve_start = time.perf_counter()
    x             = lu.solve(b)
    solve_time    = time.perf_counter() - t_solve_start

    # ------------------------------------------------------------------
    # Fill-in analysis
    # After factorisation, L and U generally have more nonzeros than A.
    # The ratio (nnz_L + nnz_U) / nnz_A quantifies how much "fill" the
    # factorisation introduced.  AMD reordering (used by default in splu)
    # minimises this.  We report it so the report can discuss sparsity.
    # ------------------------------------------------------------------
    fill = _fill_info(A, lu)

    # ------------------------------------------------------------------
    # Map solution vector back to a 2-D image
    # ------------------------------------------------------------------
    restored = system.reconstruct(x)

    return SolverResult(
        x              = x,
        restored_image = restored,
        factor_time_s  = factor_time,
        solve_time_s   = solve_time,
        fill_info      = fill,
        _lu            = lu,
        _system        = system,
    )


# ---------------------------------------------------------------------------
# Dense LU baseline — "Gaussian elimination" proxy for Phase 4 comparison
# ---------------------------------------------------------------------------

def factorise_dense(system: LaplacianSystem) -> tuple[np.ndarray, float, float, float]:
    """
    Solve the system using scipy.linalg.lu_factor / lu_solve on a DENSE
    copy of A.  This simulates re-solving from scratch (no factor reuse)
    and is the "Gaussian elimination" reference in the comparison table.

    Only usable for small systems (n ≲ 2000) because dense storage is
    O(n²).  analysis.py guards the call with a size check.

    Returns
    -------
    (x, factor_time_s, solve_time_s, condition_number)
    """
    import scipy.linalg as sla

    A_dense = system.A.toarray()
    b       = system.b

    # Condition number — expensive for large n, so only computed here
    cond = float(np.linalg.cond(A_dense))

    t0        = time.perf_counter()
    lu, piv   = sla.lu_factor(A_dense)
    fac_time  = time.perf_counter() - t0

    t0        = time.perf_counter()
    x         = sla.lu_solve((lu, piv), b)
    slv_time  = time.perf_counter() - t0

    return x, fac_time, slv_time, cond


# ---------------------------------------------------------------------------
# Condition number estimate for sparse systems (Phase 4)
# ---------------------------------------------------------------------------

def condition_number_sparse(system: LaplacianSystem) -> float:
    """
    Estimate κ(A) = ||A|| · ||A⁻¹|| using the 1-norm estimator.

    For systems too large to densify we use scipy's onenormest which
    approximates ||A⁻¹||₁ without forming A⁻¹ explicitly.
    Falls back to numpy.linalg.cond for small systems (n ≤ 500).
    """
    n = system.n
    A = system.A

    if n <= 500:
        return float(np.linalg.cond(A.toarray()))

    # 1-norm of A is cheap for sparse matrices
    norm_A   = spla.norm(A, ord=1)
    # 1-norm of A⁻¹ estimated via power iteration
    norm_Ainv = spla.onenormest(A)          # returns estimate of ||A⁻¹||₁
    return float(norm_A * norm_Ainv)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_info(A: sp.csr_matrix, lu: spla.SuperLU) -> dict:
    """
    Extract fill-in statistics from a completed splu factorisation.

    The SuperLU object exposes .L and .U as sparse matrices.
    We count their nonzeros and compare to nnz(A).
    """
    n     = A.shape[0]
    nnz_A = A.nnz

    # lu.L and lu.U are scipy sparse matrices
    nnz_L = lu.L.nnz
    nnz_U = lu.U.nnz
    nnz_LU = nnz_L + nnz_U

    fillin_factor = nnz_LU / nnz_A if nnz_A > 0 else float("inf")
    fillin_pct    = 100.0 * (nnz_LU - nnz_A) / nnz_A if nnz_A > 0 else 0.0

    return {
        "n"             : n,
        "nnz_A"         : nnz_A,
        "nnz_L"         : nnz_L,
        "nnz_U"         : nnz_U,
        "nnz_LU"        : nnz_LU,
        "fillin_factor" : round(fillin_factor, 3),
        "fillin_pct"    : round(fillin_pct, 1),
        "density_L_pct" : round(100 * nnz_L / n**2, 4) if n > 0 else 0.0,
        "density_U_pct" : round(100 * nnz_U / n**2, 4) if n > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Quick self-test  (python solver.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import urllib.request, tempfile, os
    from damage   import load_grayscale, damage_rectangle
    from laplacian import build_system

    url = ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/"
           "Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg")
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    img = load_grayscale(tmp.name, max_side=128)
    os.unlink(tmp.name)

    H, W = img.shape
    di   = damage_rectangle(img, H//4, W//4, H//4, W//4)
    sys_ = build_system(di)

    result = factorise(sys_)
    result.print_summary()

    # Also test dense path
    x_d, ft_d, st_d, cond = factorise_dense(sys_)
    print(f"\nDense LU  — factor: {ft_d*1000:.2f} ms | "
          f"solve: {st_d*1000:.2f} ms | cond(A): {cond:.2e}")

    # Visual check
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img,                   cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[1].imshow(di.damaged,            cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Damaged")
    axes[2].imshow(result.restored_image, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Restored  (residual {result.residual_norm():.1e})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("solver_check.png", dpi=120)
    plt.show()
    print("Saved solver_check.png")
