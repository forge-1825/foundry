@echo off
echo Starting Model Distillation UI...
docker-compose up -d
echo.
echo If the application started successfully, you can access it at:
echo http://localhost:3456
echo.
echo Press any key to exit...
pause > nul
