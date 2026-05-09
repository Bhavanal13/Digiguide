@echo off
echo ===================================================
echo   DIGIGUIDE - ONE-CLICK STARTUP (Fixes Connections)
echo ===================================================
echo.

echo [1/4] Refreshing USB Connection (Bypassing Wi-Fi)...
C:\Users\SK\AppData\Local\Android\Sdk\platform-tools\adb.exe reverse tcp:8000 tcp:8000
if %errorlevel% neq 0 (
    echo [WARNING] USB Bridge issue. checking devices...
    C:\Users\SK\AppData\Local\Android\Sdk\platform-tools\adb.exe devices
    echo.
    echo If device is listed above but unauthorized, check phone screen.
    echo If list is empty, reconnect USB cable.
    pause
    exit /b
)
echo [OK] USB Bridge Active.

echo.
echo [2/4] Starting Backend Server (In New Window)...
start "DigiGuide Backend" cmd /k "python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo [3/4] Waiting 5 seconds for Backend to warm up...
timeout /t 5 /nobreak >nul

echo.
echo [4/4] Starting Mobile App...
cd frontend
flutter run

echo.
echo ===================================================
echo  If App fails to launch, close this and try again!
echo ===================================================
pause
