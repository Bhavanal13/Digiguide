@echo off
echo ===================================================
echo     DIGIGUIDE - ROBUST STARTUP SCRIPT
echo ===================================================
echo.
echo [1/3] Checking USB Connection...
C:\Users\SK\AppData\Local\Android\Sdk\platform-tools\adb.exe devices
echo.
echo [2/3] Enabling USB Bridge (Bypassing Wi-Fi)...
C:\Users\SK\AppData\Local\Android\Sdk\platform-tools\adb.exe reverse tcp:8000 tcp:8000

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] USB Bridge Failed! 
    echo Please CHECK:
    echo  1. Phone is connected via USB cable
    echo  2. "USB Debugging" is ON in Developer Options
    echo  3. You accepted the "Allow USB Debugging" popup on phone
    echo.
    pause
    exit /b
)

echo [SUCCESS] USB Bridge Active!
echo.
echo [3/3] Starting App...
cd frontend
cmd /c "flutter run"
pause
