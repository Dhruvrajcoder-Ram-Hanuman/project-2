@echo off
title SENTINEL ANPR - Vehicle Intelligence Command Center
echo ====================================================================
echo   SENTINEL ANPR - Vehicle Tracking & Route Reconstruction System
echo   Starting FastAPI Server & Real-Time Surveillance Web Dashboard...
echo ====================================================================
echo.

set PYTHON_EXEC="C:\Users\amanh\AppData\Local\Programs\Python\Python311\python.exe"

if exist %PYTHON_EXEC% (
    echo Using Python 3.11 Environment: %PYTHON_EXEC%
    %PYTHON_EXEC% server.py
) else (
    echo Using default system python...
    python server.py
)

pause
