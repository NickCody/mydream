@echo off

:: Activate the Python virtual environment
CALL scripts\activate-venv.bat

:: Run the Python script with any additional arguments passed to the batch file
python -m main %* || echo Exited
