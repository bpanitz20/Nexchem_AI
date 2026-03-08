@echo off
cd /d %~dp0

echo Launching NexChem...

C:\Users\%USERNAME%\anaconda3\envs\nexchem_env\python -m streamlit run streamlit_app.py

pause