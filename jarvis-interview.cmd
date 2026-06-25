@echo off
setlocal
:: Resolve the directory where this batch file is located
set SCRIPT_DIR=%~dp0
:: Run the script using the local virtual environment Python interpreter
"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%main.py" %*
