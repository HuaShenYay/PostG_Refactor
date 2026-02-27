@echo off
chcp 65001 >nul
title 诗词推荐系统

echo ========================================
echo     诗词推荐系统 - 启动中
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] 启动前端 (npm run dev)...
start "前端服务" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo [2/2] 启动后端...
call conda run -n postg_refactor python backend\run_server.py

echo.
echo 系统已启动！
echo 前端: http://localhost:5173
echo 后端: http://localhost:5000
echo.
pause
