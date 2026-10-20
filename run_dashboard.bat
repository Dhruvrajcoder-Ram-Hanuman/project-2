@echo off
title SENTINEL ANPR - Vehicle Intelligence Command Center
echo ====================================================================
echo   SENTINEL ANPR - Vehicle Tracking & Route Reconstruction System
echo   Starting FastAPI Server & Real-Time Surveillance Web Dashboard...
echo ====================================================================
echo.

:: Detect Python 3.11 or standard python
set PY_CMD=python
if exist "C:\Users\amanh\AppData\Local\Programs\Python\Python311\python.exe" (
    set PY_CMD="C:\Users\amanh\AppData\Local\Programs\Python\Python311\python.exe"
) else (
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 (
        set PY_CMD=py -3.11
    )
)

echo [*] Launching Server using: %PY_CMD%
echo [*] Command Center URL: http://localhost:8000
echo [*] Press CTRL+C to stop the server anytime.
echo.

:: Automatically open browser after 2 seconds in background
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000"

%PY_CMD% server.py

pause
