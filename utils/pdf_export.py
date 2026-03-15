"""
PDF export utilities for Streamlit download buttons.

Two entry points:
  figures_to_pdf_bytes(figures)      – list of live matplotlib Figure objects
  image_paths_to_pdf_bytes(paths)    – list of PNG/image file paths saved to disk
"""

import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def figures_to_pdf_bytes(figures: list) -> bytes:
    """
    Combine a list of matplotlib Figure objects into a single in-memory PDF.

    Figures are NOT closed by this function; the caller retains ownership.
    Works even on figures that have already been passed to plt.close(), because
    plt.close() only removes the figure from pyplot's global registry — the
    Figure object and its canvas remain valid for savefig().

    Parameters
    ----------
    figures : list of matplotlib.figure.Figure

    Returns
    -------
    bytes : raw PDF bytes ready for st.download_button(data=...)
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def image_paths_to_pdf_bytes(paths: list) -> bytes:
    """
    Load a list of image files (PNG, JPEG, …) and combine them into a single
    in-memory PDF, one image per page.

    Uses matplotlib.pyplot.imread so no extra PIL/Pillow import is required
    (matplotlib already depends on it for non-PNG formats).

    Parameters
    ----------
    paths : list of str
        Absolute or relative paths to image files.  Non-existent paths are
        silently skipped.

    Returns
    -------
    bytes : raw PDF bytes ready for st.download_button(data=...)
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for path in paths:
            try:
                img = plt.imread(path)
            except Exception:
                continue
            h, w = img.shape[:2]
            # Use native image dimensions (100 px ≈ 1 inch)
            fig, ax = plt.subplots(figsize=(w / 100, h / 100))
            ax.imshow(img)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    buf.seek(0)
    return buf.read()
