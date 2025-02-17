@echo off
SETLOCAL EnableDelayedExpansion

:: Set error handling
SET "CPATH="
SET "LIBRARY_PATH="

:: MacOS is not applicable, hence we skip this part for Windows

:: Set project root directory (assumed to be the script's location)
SET "PROJECT_ROOT=%~dp0.."

:: Use the environment variable PYTHON_BIN if set, otherwise default to specific path
IF DEFINED PYTHON_BIN (
    SET "PYTHON_BIN=%PYTHON_BIN%"
) ELSE (
    SET "PYTHON_BIN=python3.EXE"
)

ECHO Using Python interpreter: %PYTHON_BIN%

:: Create the virtual environment in the project root
SET "VENV_PATH=%PROJECT_ROOT%\.venv"
ECHO Creating virtual environment in %VENV_PATH% ...
IF EXIST "%VENV_PATH%" (
    rem RMDIR /S /Q "%VENV_PATH%"
)
%PYTHON_BIN% -m venv "%VENV_PATH%"
IF ERRORLEVEL 1 (
    ECHO Failed to create virtual environment.
    EXIT /B 1
)

:: Activate the virtual environment (not directly possible in batch; need to invoke activation script)
CALL "%VENV_PATH%\Scripts\activate"

%VENV_PATH%\Scripts\python -m pip install --upgrade pip

:: Check if a unified requirements.txt exists
SET "REQUIREMENTS_FILE=%PROJECT_ROOT%\requirements.txt"
IF NOT EXIST "%REQUIREMENTS_FILE%" (
    ECHO Error: %REQUIREMENTS_FILE% not found.
    CALL "%VENV_PATH%\Scripts\deactivate"
    EXIT /B 1
)

ECHO Installing dependencies from %REQUIREMENTS_FILE% ...
%VENV_PATH%\Scripts\pip install -r "%REQUIREMENTS_FILE%"
IF ERRORLEVEL 1 (
    ECHO Failed to install dependencies.
    CALL "%VENV_PATH%\Scripts\deactivate"
    EXIT /B 1
)

:: OSTYPE-specific installations are limited in Windows batch, use conditional checks
:: Since Windows does not support MacOS specifics, skip to general Windows/Linux commands
ECHO ✅ Windows/Linux detected. Installing CUDA-enabled PyTorch...
%VENV_PATH%\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall

ECHO ✅ PyTorch installation complete!
:: Set PYTHONPATH to include both client and server
SET "PYTHONPATH=%PROJECT_ROOT%\client;%PROJECT_ROOT%\server;%PROJECT_ROOT%\CodeFormer"
ECHO Setting PYTHONPATH for both client/ and server/...

:: Fix CodeFormer
CALL "%VENV_PATH%\Scripts\activate"
ECHO Installing CodeFormer dependencies ...
CD "%PROJECT_ROOT%\CodeFormer"
%PYTHON_BIN% ./basicsr/setup.py install

ECHO Virtual environment setup complete.
ECHO To activate it manually, run: CALL "%VENV_PATH%\Scripts\activate"
ENDLOCAL
