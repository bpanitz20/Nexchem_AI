import sys
from pathlib import Path
import streamlit.web.cli as stcli

# Force PyInstaller to see local packages
import loaders.raman_loader
import plotting.plot_raw
import plotting.plot_regression
import plotting.plot_PCA
import models.cross_val
import models.wrappers
import models.run_loops
import models.prediction_eval
import preprocessors.aligner
import preprocessors.labeling
import preprocessors.raman_preprocess

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