@echo off
setlocal
cd /d "%~dp0"
set "KAGGLE_CONFIG_DIR=%~dp0..\tmp\visionaid-kaggle-auth"
echo Kaggle GPU connection for the VisionAid training experiment.
echo This opens the official Kaggle sign-in and authorization page.
echo Review its requested account access before approving.
echo Do not paste a password or token into the chat.
echo.
if exist "..\tmp\visionaid-kaggle-env\Scripts\kaggle.exe" (
    "..\tmp\visionaid-kaggle-env\Scripts\kaggle.exe" auth login %*
) else (
    echo The isolated Kaggle CLI environment is missing.
)
echo.
pause
