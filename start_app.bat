@echo off
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

python app.py %*
pause
