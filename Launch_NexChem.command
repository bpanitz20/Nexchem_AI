#!/bin/bash
cd "$(dirname "$0")"

/opt/anaconda3/envs/nexchem_env/bin/python3.11 -m streamlit run streamlit_app.py