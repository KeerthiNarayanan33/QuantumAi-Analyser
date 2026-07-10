@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   QuantumSentinel - Backend API Server
echo ============================================================
echo.

:: ── Check Python ─────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

:: ── Move to project root ─────────────────────────────────────
cd /d "%~dp0"

:: ── Activate virtual environment if present ──────────────────
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

:: ── Verify core packages are available ───────────────────────
echo [1/3] Verifying dependencies...
python -c "import fastapi, uvicorn, pydantic" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing missing backend packages...
    python -m pip install fastapi "uvicorn[standard]" pydantic httpx --quiet --timeout 60
)
echo       Dependencies OK.

:: ── Verify config loads correctly ────────────────────────────
echo [2/3] Verifying configuration...
python -c "from config import cfg; print('       Config OK ^— App:', cfg.app.TITLE, 'v' + cfg.app.VERSION)"
if errorlevel 1 (
    echo [ERROR] Configuration failed to load. Check config.py for errors.
    pause
    exit /b 1
)

:: ── Launch FastAPI backend ───────────────────────────────────
echo [3/3] Starting QuantumSentinel Backend API...
echo.
echo   Swagger UI  ^>  http://localhost:8000/docs
echo   ReDoc       ^>  http://localhost:8000/redoc
echo   Health      ^>  http://localhost:8000/health
echo   API Base    ^>  http://localhost:8000/api
echo.
echo   Press Ctrl+C to stop the server.
echo ============================================================
echo.

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

pause
