@echo off

:: Activate the Python virtual environment
CALL scripts\activate-venv.bat

:: Run the Python script located in the server directory with any additional arguments passed to the batch file
python server\main.py %* || echo Exited
