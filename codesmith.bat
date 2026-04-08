@echo off
REM ===========================================================
REM  codesmith - one-click launcher
REM
REM  Double-click this file (or run it from cmd / Windows Terminal)
REM  to get a menu for the codesmith agent: Web UI, REPL, info,
REM  doctor, etc. All heavy lifting lives in scripts\codesmith.ps1.
REM
REM  Usage:
REM    codesmith.bat              -> interactive menu (stays open)
REM    codesmith.bat web          -> start Web UI
REM    codesmith.bat repl         -> interactive chat in terminal
REM    codesmith.bat info         -> config + health checks
REM    codesmith.bat doctor       -> sandbox / docker / ollama probes
REM    codesmith.bat build        -> rebuild docker sandbox image
REM    codesmith.bat install      -> bootstrap venv + deps
REM    codesmith.bat chat "msg"   -> one-shot chat
REM    codesmith.bat solve "task" -> solve task with self-repair
REM ===========================================================
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\codesmith.ps1" %*
exit /b %ERRORLEVEL%
