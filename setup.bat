@echo off
echo ===================================================
echo Model Distillation UI - Setup Script
echo ===================================================
echo.

REM Check if Docker is installed
docker --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not installed or not in PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    echo and restart this script.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker Compose is not installed or not in PATH.
    echo Docker Compose should be included with Docker Desktop.
    echo Please ensure Docker Desktop is properly installed.
    pause
    exit /b 1
)

echo Docker and Docker Compose are installed. Proceeding with setup...
echo.

REM Check if NVIDIA GPU is available for CUDA
echo Checking for NVIDIA GPU...
nvidia-smi > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: NVIDIA GPU not detected or nvidia-smi not in PATH.
    echo The application will run in CPU-only mode, which may be slow.
    echo For optimal performance, an NVIDIA GPU with CUDA support is recommended.
    echo.
    set GPU_AVAILABLE=false
) else (
    echo NVIDIA GPU detected. The application will use GPU acceleration.
    echo.
    set GPU_AVAILABLE=true
)

REM Create necessary directories
echo Creating necessary directories...
mkdir data\datasets 2>nul
mkdir data\models 2>nul
mkdir data\results 2>nul
echo.

REM Build and start the Docker containers
echo Building and starting Docker containers...
echo This may take several minutes on the first run...
echo.

docker-compose up -d

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to start Docker containers.
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Setup completed successfully!
echo.
echo The Model Distillation UI is now running at:
echo http://localhost:3456
echo.
echo To stop the application, run: docker-compose down
echo To restart the application, run: docker-compose up -d
echo ===================================================
echo.

pause
