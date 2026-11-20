@echo off
title Sound Search Server
cd /d "%~dp0"
echo Starting sound search server...
python server.py
if errorlevel 1 pause
