@echo off
title Natural Presenter AI (Live Studio + Global Hotkeys)
cd /d "D:\natural-presenter-ai"

echo ============================================================
echo   Natural Presenter AI - Starting on Interactive Desktop
echo ============================================================

:: Ensure Ollama is running for Stage 1 (Gemma 3 4B)
powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 2 | Out-Null; Write-Host '[OK] Ollama is already running.' } catch { Write-Host '[INFO] Starting Ollama...'; Start-Process -FilePath \"$env:LOCALAPPDATA\Programs\Ollama\ollama.exe\" -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 2 }"

echo [INFO] Launching Web UI at http://127.0.0.1:7860 ...
echo [INFO] Global hotkeys (F9 / Alt+Space, F8 / Alt+R, F10 / Alt+S) will work across PDF, Jupyter, and OBS!
echo.
start "" "http://127.0.0.1:7860"
"C:\Users\mrbin\miniconda3\envs\presenter-ai\python.exe" app\web_ui.py
pause
