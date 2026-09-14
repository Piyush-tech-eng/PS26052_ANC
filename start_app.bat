@echo off
cd /d "%~dp0"
title PS26052 ANC - Unified Tactical Command Center
echo =======================================================================
echo   PS26052 AI-Driven Adaptive Noise Cancellation (ANC) Command Center
echo =======================================================================
echo.

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo [Notice] Virtual environment not found in .venv. Using system Python...
)

echo Starting server on:
echo   Local URL:    http://localhost:8080
echo   Direct IP:    http://127.0.0.1:8080
echo.
echo Press Ctrl+C in this terminal window to stop the server.
echo.

python app.py %*
pause
