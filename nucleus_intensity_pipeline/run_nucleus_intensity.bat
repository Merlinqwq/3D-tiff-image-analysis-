@echo off
setlocal
cd /d "%~dp0"
python nucleus_intensity.py
if errorlevel 1 pause
endlocal
