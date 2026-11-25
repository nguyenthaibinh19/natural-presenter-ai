@echo off
title Natural Presenter AI (Live Studio + Global Hotkeys)
cd /d "%~dp0"

echo ============================================================
echo   Natural Presenter AI - Starting on Interactive Desktop
echo ============================================================

:: Ensure Ollama is running for Stage 1 (Gemma 3 4B)
powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null; Write-Host '[OK] Ollama is already running.' } catch { Write-Host '[INFO] Starting Ollama...'; $ol = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'; if (Test-Path $ol) { Start-Process -FilePath $ol -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 2 } else { Write-Host '[WARN] Ollama not installed; Stage 1 will use Offline Profile Mode.' } }"

:: Locate Python from presenter-ai Conda environment (portable across machines)
set "PY_EXE="
if exist "%USERPROFILE%\miniconda3\envs\presenter-ai\python.exe" set "PY_EXE=%USERPROFILE%\miniconda3\envs\presenter-ai\python.exe"
if not defined PY_EXE if exist "%USERPROFILE%\anaconda3\envs\presenter-ai\python.exe" set "PY_EXE=%USERPROFILE%\anaconda3\envs\presenter-ai\python.exe"
if not defined PY_EXE if exist ".venv\Scripts\python.exe" set "PY_EXE=.venv\Scripts\python.exe"
if not defined PY_EXE set "PY_EXE=python"

echo [INFO] Using Python: %PY_EXE%
echo [INFO] Launching Web UI at http://127.0.0.1:7860 ...
echo [INFO] Global hotkeys (F9 / Alt+Space, F8 / Alt+R, F10 / Alt+S) work across PDF, Jupyter, and OBS!
echo.
start "" "http://127.0.0.1:7860"
"%PY_EXE%" app\web_ui.py
pause
