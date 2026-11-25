@echo off
REM Starts the FastAPI backend and the React frontend in two windows,
REM then opens the app in your browser.
cd /d "%~dp0"

start "RailSentry - Backend (port 8000)" cmd /k "C:\venvs\ailab\Scripts\python.exe -m uvicorn api:app --port 8000"
start "RailSentry - Frontend (port 5173)" cmd /k "cd frontend && npm run dev"

timeout /t 6 /nobreak >nul
start "" http://127.0.0.1:5173/
