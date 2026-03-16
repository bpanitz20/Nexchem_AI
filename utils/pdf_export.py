"""
PDF export utilities for Streamlit download buttons.

Three entry points:
  figures_to_pdf_bytes(figures)    – list of live matplotlib Figure objects → vector PDF
  pdf_paths_to_pdf_bytes(paths)    – list of file paths; swaps .png → .pdf and merges
                                     vector PDFs with pypdf (preferred, no rasterization)
  image_paths_to_pdf_bytes(paths)  – legacy fallback: embeds raster PNGs via imshow
"""

import io
from pathlib import Path
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


def pdf_paths_to_pdf_bytes(paths: list) -> bytes:
    """
    Merge pre-saved vector PDF files into one PDF without rasterizing.

    For each path in *paths*, the function looks for a sibling file with the
    same stem but a ``.pdf`` extension (e.g. ``CV_R2_PLS_EPA.png`` →
    ``CV_R2_PLS_EPA.pdf``).  Paths whose ``.pdf`` sibling does not exist are
    silently skipped.

    Parameters
    ----------
    paths : list of str
        File paths (typically ``.png``) returned by the plotting functions.

    Returns
    -------
    bytes : raw PDF bytes ready for ``st.download_button(data=...)``
    """
    from pypdf import PdfWriter
    writer = PdfWriter()
    for path in paths:
        pdf_path = Path(path).with_suffix(".pdf")
        if pdf_path.exists():
            writer.append(str(pdf_path))
    buf = io.BytesIO()
    writer.write(buf)
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
