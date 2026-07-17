@echo off
:: ============================================================
::  launch-web.bat — Levanta la UI web de TripoSR (CPU)
::  Valida el entorno y abre Gradio en http://127.0.0.1:7860
:: ============================================================

:: --- Configuracion centralizada ------------------------------
set "VENV_PY=%USERPROFILE%\venvs\triposr\Scripts\python.exe"
set "APP_DIR=%~dp0"
set "APP_URL=http://127.0.0.1:7860"
:: -------------------------------------------------------------

if not exist "%VENV_PY%" (
    echo [ERROR] No se encontro el venv en: %VENV_PY%
    echo         Revisa la seccion "Reinstalacion del entorno" del README.
    pause
    exit /b 1
)

if not exist "%APP_DIR%gradio_app.py" (
    echo [ERROR] No se encontro gradio_app.py en: %APP_DIR%
    pause
    exit /b 1
)

echo [INFO] Iniciando TripoSR Web UI...
echo [INFO] Abre %APP_URL% cuando aparezca "Running on local URL".
echo [INFO] Deten el servidor con Ctrl+C.
echo.

cd /d "%APP_DIR%"
"%VENV_PY%" gradio_app.py

pause
