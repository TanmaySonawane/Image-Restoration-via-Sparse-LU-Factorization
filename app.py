"""
app.py  —  Gradio UI for Hugging Face Spaces deployment
=========================================================
Entry point for the HF Space.  Hugging Face looks for app.py at the
repository root, reads the sdk: gradio line in README.md, installs
requirements.txt, and runs this file.

The UI exposes three controls:
  1. Image upload (or use the bundled sample)
  2. Damage type selector: Rectangle block | Diagonal scratch
  3. Damage intensity slider (controls block size / scratch thickness)

Outputs:
  - Original image
  - Damaged image
  - Restored image
  - Residual heatmap
  - A text summary (n unknowns, κ(A), residual norm, solve time)

All heavy computation is done by the same src/ modules used locally,
so the HF result is byte-for-byte identical to a local run.

Folder layout expected on HF:
  app.py              ← this file (HF entry point)
  requirements.txt
  src/
    damage.py
    laplacian.py
    solver.py
    analysis.py
    pipeline.py
  assets/
    sample_image.png
"""

from __future__ import annotations

import sys
import io
import textwrap
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required on HF
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Make src/ importable whether app.py is run from root or src/
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent
_src  = _root / "src"
for p in (_root, _src):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from damage    import load_grayscale, damage_rectangle, damage_diagonal_scratch
from laplacian import build_system
from solver    import factorise, condition_number_sparse


# ---------------------------------------------------------------------------
# Core processing function  (called by Gradio on every submission)
# ---------------------------------------------------------------------------

def process(
    pil_image   : Image.Image | None,
    damage_type : str,
    intensity   : int,
    max_side    : int,
) -> tuple[
    np.ndarray,   # original (for gr.Image)
    np.ndarray,   # damaged
    np.ndarray,   # restored
    np.ndarray,   # residual heatmap (matplotlib figure as numpy RGBA)
    str,          # text summary
]:
    """
    Main processing callback wired to the Gradio interface.

    Parameters
    ----------
    pil_image   : uploaded PIL image, or None (falls back to sample)
    damage_type : "Rectangle block" | "Diagonal scratch"
    intensity   : 1–100 slider value
                    rect   → block side  = round(min(H,W) * intensity/200)
                    scratch → thickness  = max(1, round(intensity / 15))
    max_side    : image is downscaled so longest side ≤ max_side

    Returns
    -------
    Five values consumed by Gradio output components.
    """
    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    if pil_image is None:
        sample = _root / "assets" / "sample_image.png"
        if not sample.exists():
            _download_sample(sample)
        pil_image = Image.open(sample)

    # Convert PIL → float64 greyscale [0,1]
    img = _pil_to_float(pil_image, max_side=max_side)
    H, W = img.shape

    # ------------------------------------------------------------------
    # 2. Apply damage  (Phase 1)
    # ------------------------------------------------------------------
    if damage_type == "Rectangle block":
        side = max(4, round(min(H, W) * intensity / 200))
        r0   = max(1, (H - side) // 2)
        c0   = max(1, (W - side) // 2)
        # Ensure the block stays 1 pixel away from every border
        side_r = min(side, H - r0 - 1)
        side_c = min(side, W - c0 - 1)
        di = damage_rectangle(img, r0, c0, side_r, side_c)
    else:
        thickness = max(1, round(intensity / 15))
        di = damage_diagonal_scratch(img, thickness=thickness)

    # ------------------------------------------------------------------
    # 3. Build Laplacian system  (Phase 2)
    # ------------------------------------------------------------------
    try:
        system = build_system(di)
    except ValueError as exc:
        # Edge case: intensity so low that no pixels are masked
        return (
            _to_uint8(img), _to_uint8(img), _to_uint8(img),
            _blank_heatmap(),
            f"Error: {exc}\nTry increasing the damage intensity.",
        )

    # ------------------------------------------------------------------
    # 4. Sparse LU solve  (Phase 3)
    # ------------------------------------------------------------------
    result = factorise(system)

    # ------------------------------------------------------------------
    # 5. Condition number  (expensive for large n — cap at n ≤ 3000)
    # ------------------------------------------------------------------
    cond_str = "skipped (n > 3 000)"
    if system.n <= 3000:
        cond = condition_number_sparse(system)
        cond_str = f"{cond:.3e}"

    # ------------------------------------------------------------------
    # 6. Residual heatmap  (Phase 5)
    # ------------------------------------------------------------------
    heatmap_arr = _make_heatmap(di, result)

    # ------------------------------------------------------------------
    # 7. Text summary
    # ------------------------------------------------------------------
    summary = textwrap.dedent(f"""
        ── System info ──────────────────────────────
          Image size        : {H} × {W}
          Missing pixels    : {di.n_missing:,}  ({100*di.n_missing/(H*W):.1f}%)
          Matrix A size     : {system.n:,} × {system.n:,}
          Nonzeros in A     : {system.A.nnz:,}
          Density of A      : {100*system.A.nnz/system.n**2:.4f}%

        ── Fill-in (L + U vs A) ─────────────────────
          nnz in L          : {result.fill_info['nnz_L']:,}
          nnz in U          : {result.fill_info['nnz_U']:,}
          Fill-in factor    : {result.fill_info['fillin_factor']}×

        ── Solve ────────────────────────────────────
          Factorisation     : {result.factor_time_s*1000:.2f} ms
          Solve time        : {result.solve_time_s*1000:.2f} ms
          Relative residual : {result.residual_norm():.2e}
          Condition κ(A)    : {cond_str}
    """).strip()

    return (
        _to_uint8(img),
        _to_uint8(di.damaged),
        _to_uint8(result.restored_image),
        heatmap_arr,
        summary,
    )


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""

    with gr.Blocks(
        title="Image Restoration via Sparse LU Factorisation",
        theme=gr.themes.Soft(),
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.Markdown(
            """
            # Image Restoration via Sparse LU Factorisation
            **DSC 301 · Linear Algebra project · UMass Dartmouth**

            Upload a grayscale (or colour) image, choose a damage type and
            intensity, then click **Restore**.  Missing pixels are recovered
            by solving the discrete Laplace equation
            **4u[i,j] − u[i−1,j] − u[i+1,j] − u[i,j−1] − u[i,j+1] = 0**
            as a sparse linear system **Ax = b** using LU factorisation.

            > Source code and report: [GitHub](https://github.com)
            *(replace with your repo link)*
            """
        )

        # ── Inputs ──────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Inputs")
                img_input = gr.Image(
                    label   = "Upload image  (leave blank to use built-in sample)",
                    type    = "pil",
                    sources = ["upload", "clipboard"],
                )
                damage_type = gr.Radio(
                    choices = ["Rectangle block", "Diagonal scratch"],
                    value   = "Rectangle block",
                    label   = "Damage type",
                )
                intensity = gr.Slider(
                    minimum = 5,
                    maximum = 80,
                    value   = 30,
                    step    = 1,
                    label   = "Damage intensity  (block size / scratch thickness)",
                )
                max_side = gr.Slider(
                    minimum = 64,
                    maximum = 512,
                    value   = 200,
                    step    = 32,
                    label   = "Max image side (px)  —  larger = slower solve",
                )
                run_btn = gr.Button("Restore", variant="primary")

        # ── Outputs ─────────────────────────────────────────────────────
        gr.Markdown("### Results")
        with gr.Row():
            out_original = gr.Image(label="Original",       type="numpy")
            out_damaged  = gr.Image(label="Damaged",        type="numpy")
            out_restored = gr.Image(label="Restored",       type="numpy")
            out_heatmap  = gr.Image(label="Residual error", type="numpy")

        out_summary = gr.Textbox(
            label    = "Solver summary",
            lines    = 16,
            max_lines= 20,
        )

        # ── Examples ────────────────────────────────────────────────────
        gr.Markdown("### Try an example")
        gr.Examples(
            examples=[
                [None, "Rectangle block",  20, 200],
                [None, "Rectangle block",  55, 200],
                [None, "Diagonal scratch", 40, 200],
            ],
            inputs=[img_input, damage_type, intensity, max_side],
            label="Click a row to pre-fill the inputs, then click Restore",
        )

        # ── Method description (collapsible) ────────────────────────────
        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                """
                **Step 1 — Damage simulation**
                Selected pixels are zeroed out and recorded in a binary mask.

                **Step 2 — Laplacian system (Ax = b)**
                Each missing pixel u[i,j] satisfies the discrete Laplace equation:
                the pixel equals the average of its four neighbours.
                Rearranging gives one equation per missing pixel.
                Known (boundary) pixels move to the right-hand side b.
                The result is a sparse n×n system where n = number of missing pixels.

                **Step 3 — Sparse LU factorisation**
                `scipy.sparse.linalg.splu` factors A = P·L·U using AMD reordering
                (minimises fill-in) and partial pivoting (numerical stability).
                Two triangular solves recover x — the unknown pixel values.

                **Step 4 — Fill-in analysis**
                L and U are denser than A.  The fill-in factor measures how much
                extra memory the factorisation requires vs the original matrix.
                AMD reordering keeps this factor small.

                **Step 5 — Results**
                The solution vector x is written back into the image.
                The residual heatmap shows |restored − original| per pixel
                inside the damaged region (only meaningful when the original
                is known, i.e. in this controlled experiment).
                """
            )

        # ── Wire up ─────────────────────────────────────────────────────
        run_btn.click(
            fn      = process,
            inputs  = [img_input, damage_type, intensity, max_side],
            outputs = [out_original, out_damaged, out_restored,
                       out_heatmap, out_summary],
        )

    return demo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_float(pil_img: Image.Image, max_side: int = 256) -> np.ndarray:
    """Convert PIL image → float64 greyscale [0, 1], downscaled if needed."""
    img = pil_img.convert("L")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert float64 [0,1] to uint8 [0,255] RGB for Gradio display."""
    clipped = np.clip(arr, 0.0, 1.0)
    grey    = (clipped * 255).astype(np.uint8)
    # Gradio gr.Image expects HxWx3
    return np.stack([grey, grey, grey], axis=-1)


def _make_heatmap(di, result) -> np.ndarray:
    """
    Render the per-pixel residual as a hot-colourmap figure.
    Returns a uint8 HxWx3 array.
    """
    err        = np.abs(result.restored_image - di.original)
    err_masked = np.where(di.mask, err, np.nan)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(err_masked, cmap="hot", vmin=0, vmax=0.1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="|restored − original|")
    ax.set_title("Reconstruction error\n(masked region only)", fontsize=9)
    ax.axis("off")
    plt.tight_layout()

    # Render to numpy array via in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    pil_fig = Image.open(buf).convert("RGB")
    return np.asarray(pil_fig, dtype=np.uint8)


def _blank_heatmap() -> np.ndarray:
    """Return a plain grey placeholder when no heatmap can be computed."""
    arr = np.full((200, 200, 3), 200, dtype=np.uint8)
    return arr


def _download_sample(dest: Path) -> None:
    """Download a public-domain fallback image if assets/ is empty."""
    import urllib.request
    url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/"
        "Camponotus_flavomarginatus_ant.jpg/"
        "320px-Camponotus_flavomarginatus_ant.jpg"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception:
        # If the download fails on HF, create a synthetic gradient image
        synthetic = (np.linspace(0, 255, 200 * 200)
                       .reshape(200, 200)
                       .astype(np.uint8))
        Image.fromarray(synthetic).save(dest)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    # share=False for local runs; HF ignores this argument entirely
    demo.launch(share=False)
