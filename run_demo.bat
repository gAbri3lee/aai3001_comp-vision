@echo off
setlocal
cd /d "%~dp0"
echo VisionAid demo - open http://127.0.0.1:5001 in your browser.
echo Keep this window open. Press Ctrl+C here to stop the app.
echo.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" src/webapp.py --no-depth --port 5001 %*
) else if exist "..\tmp\visionaid-validation-env\Scripts\python.exe" (
    "..\tmp\visionaid-validation-env\Scripts\python.exe" -c "import site,runpy,sys; site.addsitedir(r'..\tmp\visionaid-validation-deps'); sys.path.insert(0,'src'); sys.argv=['src/webapp.py','--no-depth','--port','5001']+sys.argv[1:]; runpy.run_path('src/webapp.py',run_name='__main__')" %*
) else (
    echo Python environment missing. Follow the Windows setup in README.md.
)
echo.
pause
