@echo off
echo ==========================================
echo AI Question Answering System
echo ==========================================
echo.
python scripts\download_dataset.py
if errorlevel 1 goto failed
python scripts\build_index.py
if errorlevel 1 goto failed
python scripts\train_rnn.py
if errorlevel 1 goto failed
python backend\app.py
goto end
:failed
echo.
echo Setup failed. Read the error above.
pause
:end
