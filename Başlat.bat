@echo off
title Cicek Tanima Sistemi

cd /d "%~dp0"

call "venv\Scripts\activate.bat"

python -m streamlit run "cicek_app.py" --server.port 8502

pause