@echo off
title RMBG 2.0 Background Remover Local API Service
echo ======================================================================
echo          RMBG 2.0 Background Remover - Local API Launcher
echo ======================================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python 3.10 or 3.11 and check "Add Python to PATH" during setup.
    echo.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo [INFO] Creating a Python virtual environment (venv) in local folder...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        echo.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
    echo.
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate
echo.

:: Install dependencies
echo [INFO] Installing/Updating required CPU-only dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    echo.
    pause
    exit /b 1
)
echo [SUCCESS] Dependencies verified.
echo.

:: Launch the HTML frontend in default browser
echo [INFO] Opening HTML frontend in default browser...
start index.html

:: Run the FastAPI backend service
echo [INFO] Starting local FastAPI backend server on http://localhost:8000...
echo.
uvicorn app:app --host 127.0.0.1 --port 8000

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] FastAPI backend stopped unexpectedly.
    pause
)
