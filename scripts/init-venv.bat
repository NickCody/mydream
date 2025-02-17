@echo off
REM Enable delayed variable expansion (if needed)
setlocal enabledelayedexpansion

REM ----------------------------------------------------------------------
REM This script is intended for Windows.
REM The original script checked for Darwin/macOS; here we assume Windows.
set "DARWIN_FOUND=0"

REM ----------------------------------------------------------------------
REM Set the project root directory (assumed to be the parent directory of this script)
pushd "%~dp0\.."
set "PROJECT_ROOT=%CD%"
popd

echo Using Python interpreter:
where python3 2>nul || where python

REM ----------------------------------------------------------------------
REM Create the virtual environment in the project root
set "VENV_PATH=%PROJECT_ROOT%\.venv"
echo Creating virtual environment in %VENV_PATH% ...
if exist "%VENV_PATH%" (
    rmdir /s /q "%VENV_PATH%"
)

python -m venv "%VENV_PATH%"
if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
)

REM ----------------------------------------------------------------------
REM Activate the virtual environment
echo Activating virtual environment ...
call "%VENV_PATH%\Scripts\activate.bat"

REM ----------------------------------------------------------------------
REM Upgrade pip and install numpy with version constraint
echo Upgrading pip ...
python -m pip install --upgrade pip

echo Installing numpy (version less than 2)...
python -m pip install "numpy<2"

REM ----------------------------------------------------------------------
REM Check if a unified requirements.txt exists
set "REQUIREMENTS_FILE=%PROJECT_ROOT%\requirements.txt"
if not exist "%REQUIREMENTS_FILE%" (
    echo Error: %REQUIREMENTS_FILE% not found.
    exit /b 1
)

echo Installing dependencies from %REQUIREMENTS_FILE% ...
python -m pip install -c constraints.txt -r "%REQUIREMENTS_FILE%"
if errorlevel 1 (
    echo Failed to install dependencies.
    exit /b 1
)

REM ----------------------------------------------------------------------
REM Install additional dependencies depending on OS.
if "%DARWIN_FOUND%"=="1" (
    python -m pip install -c constraints.txt -r gui-requirements.txt
) else (
    python -m pip install -c constraints.txt transformers "diffusers[torch]" tf-keras==2.17.0 accelerate
    python -m pip install -c constraints.txt torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall
)

echo PyTorch installation complete!

REM ----------------------------------------------------------------------
REM Set PYTHONPATH to include both client and server directories.
echo Setting PYTHONPATH for client, server, and CodeFormer...
set "PYTHONPATH=%PROJECT_ROOT%\client;%PROJECT_ROOT%\server;%PROJECT_ROOT%\CodeFormer"

REM ----------------------------------------------------------------------
REM Fix CodeFormer: Activate the venv (again) and install CodeFormer dependencies.
call "%VENV_PATH%\Scripts\activate.bat"
echo Installing CodeFormer dependencies ...
cd "%PROJECT_ROOT%\CodeFormer"
REM Uncomment the following line if CodeFormer installation is needed:
python3 basicsr\setup.py install

echo Virtual environment setup complete.
echo To activate it manually, run: call "%VENV_PATH%\Scripts\activate.bat"
pause