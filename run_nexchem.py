import sys
from pathlib import Path
import streamlit.web.cli as stcli

# Force PyInstaller to see local packages
import loaders.raman_loader
import plotting.plot_raw
import plotting.plot_regression
import plotting.plot_PCA
import plotting.plot_classifier
import models.cross_val
import models.wrappers
import models.classification_wrappers
import models.run_loops
import models.prediction_eval
import models.vip
import models.block_selection
import models.selectors.vip
import models.selectors.block
import preprocessors.aligner
import preprocessors.labeling
import preprocessors.raman_preprocess
import preprocessors.transforms
import utils.pdf_export

def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        return str(Path(sys._MEIPASS) / relative_path)
    return str(Path(__file__).parent / relative_path)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.argv = [
        "streamlit",
        "run",
        resource_path("streamlit_app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())