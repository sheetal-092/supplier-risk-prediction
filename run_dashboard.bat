@echo off
REM Launch SUPPLIER RISK PREDICTION locally (Windows).
REM Creates the virtual environment on first run.
cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo Creating virtual environment ...
    where uv >nul 2>nul
    IF %ERRORLEVEL%==0 (
        uv venv --python 3.12 .venv && uv pip install --python .venv\Scripts\python.exe -r requirements.txt
    ) ELSE (
        py -3.12 -m venv .venv && .venv\Scripts\python.exe -m pip install -r requirements.txt
    )
)

IF NOT EXIST "models\model_registry.json" (
    echo Training models ...
    .venv\Scripts\python.exe scripts\train_models.py
)

echo Starting dashboard at http://localhost:8501  (Ctrl+C to stop)
.venv\Scripts\python.exe -m streamlit run app.py
