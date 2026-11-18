@echo off
setlocal
cd /d "%~dp0"
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
  py -3.12 launcher.py
) else (
  python launcher.py
)
if errorlevel 1 (
  echo.
  echo Launch failed. Install Python 3.12 64-bit, then try again.
  pause
)
