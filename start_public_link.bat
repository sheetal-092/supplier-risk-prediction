@echo off
REM Starts the dashboard AND a temporary public link (Cloudflare quick tunnel).
REM The link works from any laptop/phone while this window stays open.
cd /d "%~dp0"
IF NOT EXIST ".venv\Scripts\python.exe" (
    echo Virtual environment missing - run run_dashboard.bat once first.
    pause & exit /b 1
)
IF NOT EXIST "cloudflared.exe" (
    echo Downloading cloudflared ...
    curl -L -o cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
)
start "SUPPLIER RISK PREDICTION - app" cmd /k .venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.headless true
timeout /t 8 >nul
echo.
echo ==== Your public link appears below (https://....trycloudflare.com) ====
echo ==== Keep this window open. Close it to stop sharing.                ====
echo.
cloudflared.exe tunnel --url http://localhost:8501 --no-autoupdate
