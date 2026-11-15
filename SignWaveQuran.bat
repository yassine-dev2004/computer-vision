@echo off
rem Quran revision with the Arabic finger alphabet. Options are passed on, e.g.  SignWaveQuran.bat --surah 112
cd /d "%~dp0"
set PY=python
if exist "%USERPROFILE%\venvs\signwave\Scripts\python.exe" set PY="%USERPROFILE%\venvs\signwave\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set PY=".venv\Scripts\python.exe"
%PY% -m arabic.live %*
if errorlevel 1 pause
