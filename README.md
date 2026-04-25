---
title: Image Restoration Sparse LU
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Image Restoration via Sparse LU Factorization

**DSC 301 — Linear Algebra · UMass Dartmouth**

A mathematical image inpainting tool that recovers missing or damaged pixels
by solving the discrete Laplace equation as a sparse linear system **Ax = b**
using LU factorization.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/T22S/image-restoration-sparse-lu)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/TanmaySonawane/Image-Restoration-via-Sparse-LU-Factorization)

---

## How it works

Every missing pixel `u[i,j]` satisfies the discrete Laplace equation:

```
4·u[i,j] − u[i−1,j] − u[i+1,j] − u[i,j−1] − u[i,j+1] = 0
```

This says the pixel should equal the average of its four neighbours —
a natural smoothness condition.  Applying this to every missing pixel
produces a sparse linear system:

```
A · x = b
```

where `x` is the vector of unknown pixel values, `A` encodes the
Laplacian stencil (≤ 5 nonzeros per row), and `b` collects contributions
from known boundary pixels.  The system is solved with
`scipy.sparse.linalg.splu` — a direct sparse LU factorizer that uses
AMD reordering to minimize fill-in and partial pivoting for stability.

---

## Project structure

```
Image-Restoration-via-Sparse-LU-Factorization/
│
├── app.py                  # Gradio UI — Hugging Face entry point
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore
│
├── src/
│   ├── damage.py           # Phase 1 — image loading + damage simulation
│   ├── laplacian.py        # Phase 2 — sparse Laplacian system Ax=b
│   ├── solver.py           # Phase 3 — sparse LU factorization + solve
│   ├── analysis.py         # Phase 4 — timing + condition number comparison
│   └── pipeline.py         # Phases 1–5 orchestrator (local runner)
│
├── assets/
│   └── sample_image.png    # Bundled demo image
│
└── results/                # Generated figures + CSV (not committed)
```

---

## Running locally

**Requirements:** Python 3.10+, Windows / macOS / Linux

```bash
# 1. Clone the repo
git clone https://github.com/TanmaySonawane/Image-Restoration-via-Sparse-LU-Factorization.git
cd Image-Restoration-via-Sparse-LU-Factorization

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (downloads a demo image automatically)
python src/pipeline.py

# 5. Or launch the Gradio UI locally
python app.py
```

Optional pipeline flags:
```bash
python src/pipeline.py --image path/to/your/image.png
python src/pipeline.py --max-side 300
python src/pipeline.py --sizes 15 35 55
python src/pipeline.py --scratch      # also run diagonal scratch demo
```

Results (figures + CSV table) are saved to `results/`.

---

## Module descriptions

| File | Phase | Role |
|------|-------|------|
| `src/damage.py` | 1 | Load grayscale image; apply rectangular block or diagonal scratch damage; return binary mask |
| `src/laplacian.py` | 2 | Build sparse CSR matrix A and vector b from the discrete Laplace equation |
| `src/solver.py` | 3 | Factorise A = PLU via `splu`; solve; report fill-in; expose reuse API |
| `src/analysis.py` | 4 | Compare sparse LU vs dense LU across three damage sizes; generate all four figures + CSV |
| `src/pipeline.py` | 1–5 | Orchestrates all phases; CLI entry point for local runs |
| `app.py` | — | Gradio UI; Hugging Face Spaces entry point |

---

## Key results

- **Residual** `‖Ax − b‖ / ‖b‖` consistently at machine precision (~10⁻¹⁵)
- **Fill-in factor** 3–8× depending on damage geometry (AMD reordering applied)
- **LU reuse** reduces per-solve cost to < 0.1 ms vs 1–10 ms for full refactorization
- **Condition number** κ(A) grows with damage area, explaining accuracy limits on large holes

---

## Mathematical background

| Concept | Where used |
|---------|-----------|
| Discrete Laplace equation | `laplacian.py` — derives each row of A |
| Sparse matrix (CSR format) | `laplacian.py` — stores A with ≤ 5 nonzeros/row |
| LU factorization (A = PLU) | `solver.py` — `scipy.sparse.linalg.splu` |
| AMD reordering | `solver.py` — minimizes fill-in before factorizing |
| Partial pivoting | `solver.py` — ensures numerical stability |
| Condition number κ(A) | `solver.py`, `analysis.py` — measures system sensitivity |
| Forward / back substitution | `solver.py` — triangular solves after factorization |

---

## References

1. Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
2. Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
3. Bertalmío, M. et al. (2000). Image inpainting. *SIGGRAPH 2000*.
4. SciPy documentation — `scipy.sparse.linalg.splu`:
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html

---

## Author

Tanmay Sonawane · UMass Dartmouth · DSC 301
