@echo off
echo ============================================================
echo   QuantumSentinel - Quantum Analytics Platform
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

:: Install dependencies if needed
echo [1/3] Checking dependencies...
pip install -r requirements.txt --quiet

:: Train models
echo [2/3] Training models (this may take a few minutes)...
python src/train.py

:: Launch dashboard
echo [3/3] Launching dashboard...
echo.
echo Open your browser at: http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run dashboard/app.py
