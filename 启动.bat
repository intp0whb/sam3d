@echo off
echo ============================================
echo SAM-3D 整合包 - 正在启动后端服务...
echo ============================================
echo.

start cmd /k python_embedded\python.exe server.py

echo 等待后端服务启动...
echo 请稍候，正在检测服务状态...
echo.

:wait_for_server
timeout /t 2 >nul

:: 检测服务器是否就绪
curl -s http://localhost:5001/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo 后端服务已启动完成！
    echo ============================================
    echo.
    echo 正在打开浏览器...
    start http://localhost:5001
    echo.
    echo 后端服务窗口已打开，请勿关闭。
    pause
    exit /b
)

echo 服务启动中... 请稍候
goto wait_for_server