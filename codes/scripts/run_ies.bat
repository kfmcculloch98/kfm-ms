@echo off
setlocal

set PEST_DIR=C:\Python\Personal\kfm-ms\codes\pest
set IES_EXE=C:\Python\Personal\kfm-ms\codes\binaries\PESTPP\windows\pestpp-ies.exe
set PST_NAME=inversion_level_1.pst

cd /d "%PEST_DIR%"

echo === starting pestpp-ies (version 2 pst) ===
echo.

:: PESTPP-IES performs its own validation immediately
"%IES_EXE%" %PST_NAME%

if %ERRORLEVEL% NEQ 0 goto :error
echo.
echo === inversion complete ===
goto :end

:error
echo.
echo [error] PESTPP-IES failed. check the PEST records (*.rec) in the PEST folder.
pause
:end
pause