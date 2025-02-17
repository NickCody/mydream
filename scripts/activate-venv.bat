@echo off

:: Set project root directory (adjusted for scripts/ location)
SET "PROJECT_ROOT=%~dp0.."

:: Path to virtual environment
SET "VENV_PATH=%PROJECT_ROOT%\.venv"

:: Check if the virtual environment exists
IF NOT EXIST "%VENV_PATH%" (
  echo Error: Virtual environment not found at %VENV_PATH%
  echo Run .\scripts\init-venv.bat to create it.
  exit /b 1
)

:: Activate the virtual environment
echo Activating virtual environment...
CALL "%VENV_PATH%\Scripts\activate"

:: Set PYTHONPATH to include client, server, and CodeFormer directories
SET "PYTHONPATH=%PROJECT_ROOT%\client;%PROJECT_ROOT%\server;%PROJECT_ROOT%\CodeFormer"
echo PYTHONPATH set to: %PYTHONPATH%
echo Virtual environment activated. 🚀
