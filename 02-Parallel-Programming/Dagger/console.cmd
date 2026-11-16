@echo off
rem Run the Unix SSH/PTY console in WSL; pass all options through unchanged.
where wsl.exe >nul 2>nul
if errorlevel 1 (
    echo The console needs Windows Subsystem for Linux with Python 3.9+ and OpenSSH.
    echo Install WSL with: wsl --install
    exit /b 1
)
wsl.exe --cd "%~dp0." --exec /bin/sh ./console %*
exit /b %errorlevel%
