@echo off
TITLE VoxAuth Neural Engine Launcher
color 0B

echo ===================================================
echo        VoxAuth Biometric App Launcher
echo ===================================================
echo.

:: 1. Check Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    color 0C
    echo ---------------------------------------------------
    echo [ERROR] Python is not installed. 
    echo Please install Python 3.10 or newer from python.org!
    echo Ensure you check "Add python.exe to PATH" during installation.
    echo ---------------------------------------------------
    pause
    exit /b
)

:: 2. Setup venv
echo [2/4] Verifying isolated workspace (.venv)...
IF NOT EXIST .venv (
    echo         Setting up a fresh local Python environment...
    python -m venv .venv
)

:: 3. Activate venv
call .venv\Scripts\activate.bat

:: 4. Install dependencies efficiently (CPU Torch takes way less space!)
echo [3/4] Ensuring Neural AI and Libraries are installed...
python -c "import torch" >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo         Installing Core AI Backend (PyTorch CPU - Lightweight Variant)...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo         Configuring Audio Toolkit and UI requirements...
pip install -r requirements.txt --quiet

:: 5. Launch
echo [4/4] Starting the App...
echo.
echo Please keep this terminal window open while using VoxAuth.
echo.
streamlit run app.py

pause
