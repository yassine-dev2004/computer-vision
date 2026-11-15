@echo off
rem Launch the SignWaveAI webcam demo. Extra options are passed on, e.g.  SignWaveAI.bat --no-cnn
cd /d "%~dp0"
set PY=python
if exist "%USERPROFILE%\venvs\signwave\Scripts\python.exe" set PY="%USERPROFILE%\venvs\signwave\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set PY=".venv\Scripts\python.exe"
%PY% -m recognition.live %*
if errorlevel 1 pause
