@echo off
setlocal

cd /d "%~dp0"

echo [1/2] Starting Docker services (Virtuoso + Qdrant)...
docker compose up -d
if errorlevel 1 goto :error

echo [2/2] Launching Streamlit frontend...
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run ama-kbqa-frontend %*
) else (
    ama-kbqa-frontend %*
)
goto :eof

:error
echo Failed to start Docker services. Is Docker Desktop running?
exit /b 1
