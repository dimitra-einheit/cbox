@echo off
rem dronemap launcher for Windows. Uses the toolkit's own Python environment in .venv
rem (see setup\SETUP_WINDOWS.md).
setlocal
set "HERE=%~dp0"
if not exist "%HERE%.venv\Scripts\python.exe" (
  echo dronemap is not set up yet. Follow setup\SETUP_WINDOWS.md
  exit /b 1
)
set "PYTHONPATH=%HERE%src;%PYTHONPATH%"
"%HERE%.venv\Scripts\python.exe" -m dronemap %*
exit /b %ERRORLEVEL%
