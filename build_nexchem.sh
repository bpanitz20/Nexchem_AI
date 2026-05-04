#!/bin/bash

cd "$(dirname "$0")"

rm -rf build dist *.spec __pycache__

pyinstaller --onefile \
  --paths . \
  --copy-metadata streamlit \
  --collect-all streamlit \
  --collect-all matplotlib \
  --collect-submodules loaders \
  --collect-submodules plotting \
  --collect-submodules models \
  --collect-submodules preprocessors \
  --collect-submodules utils \
  --add-data "streamlit_app.py:." \
  run_nexchem.py
