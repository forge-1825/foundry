@echo off
echo Stopping Model Distillation UI...
docker-compose down
echo.
echo Application stopped.
echo.
echo Press any key to exit...
pause > nul
