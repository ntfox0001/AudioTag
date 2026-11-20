@echo off
title AudioTag Setup
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ first.
    pause
    exit /b 1
)

echo [1/4] Installing dependencies ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)

echo [2/4] Downloading CLAP model ^(about 1.8 GB^) ...
python download_model.py laion/clap-htsat-fused
if errorlevel 1 (
    echo [ERROR] CLAP model download failed.
    pause
    exit /b 1
)

echo [3/4] Downloading translation model ...
python download_model.py Helsinki-NLP/opus-mt-zh-en
if errorlevel 1 (
    echo [ERROR] translation model download failed.
    pause
    exit /b 1
)

echo [4/4] Setup finished.
echo.
echo Next steps:
echo   1. Put your audio files in a folder ^(or run: python download_kenney.py^)
echo   2. Double-click "%~dp0start_audio_tag.bat" to start the server
echo   3. Open the "Index" tab in the web page and build the index
echo.
pause
