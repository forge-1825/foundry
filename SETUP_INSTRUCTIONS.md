# Model Distillation UI - Setup Instructions

This document provides detailed instructions for setting up and running the Model Distillation UI application.

## System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux
- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **Disk Space**: 10GB minimum for the application, plus additional space for models
- **GPU**: NVIDIA GPU with CUDA support recommended for optimal performance
- **Software**: Docker Desktop (includes Docker Engine and Docker Compose)

## Quick Start

For Windows users, the easiest way to get started is to run the included `setup.bat` script:

1. Double-click `setup.bat`
2. Follow the on-screen instructions
3. Once setup completes, access the UI at http://localhost:3456

For macOS/Linux users, or for manual setup, follow the detailed instructions below.

## Detailed Setup Instructions

### 1. Install Docker

If you don't already have Docker installed:

- **Windows/macOS**: Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Follow the [Docker Engine installation instructions](https://docs.docker.com/engine/install/) for your distribution

After installation, verify Docker is working by running:
```
docker --version
docker-compose --version
```

### 2. Configure Docker Resources

For optimal performance:

1. Open Docker Desktop
2. Go to Settings > Resources
3. Allocate at least 8GB of RAM and 4 CPUs
4. If you have an NVIDIA GPU, ensure the "Enable GPU support" option is checked

### 3. Start the Application

From the application directory, run:

```
docker-compose up -d
```

This will:
- Build the necessary Docker images (first run only)
- Start the frontend and backend containers
- Run the application in the background

### 4. Access the UI

Once the containers are running, access the web interface at:

```
http://localhost:3456
```

## Using Model Containers

The application is designed to work with LLM models running in Docker containers. For optimal performance, you should run:

1. A Llama 3 teacher model container
2. A Phi-3 student model container

### Setting Up Model Containers

To run the Llama 3 teacher model:

```
docker run -d --gpus all -p 8000:8000 --name llama3_teacher_vllm ghcr.io/meta-llama/llama3:latest
```

To run the Phi-3 student model:

```
docker run -d --gpus all -p 8002:8000 --name phi3_vllm microsoft/phi-3-mini-4k-instruct:latest
```

## Troubleshooting

### Common Issues

1. **Docker containers fail to start**
   - Check Docker logs: `docker logs model-distillation-ui-frontend-1`
   - Ensure ports 3456 and 7433 are not in use by other applications

2. **Cannot connect to model containers**
   - Verify model containers are running: `docker ps`
   - Check model container logs: `docker logs llama3_teacher_vllm`

3. **UI shows but features don't work**
   - Check backend logs: `docker logs model-distillation-ui-backend-1`
   - Ensure all required model containers are running

### Getting Help

If you encounter issues not covered here, please:

1. Check the detailed logs in the Docker containers
2. Refer to the documentation in the `docs` directory
3. Contact the development team with specific error messages and logs

## Stopping the Application

To stop the application:

```
docker-compose down
```

To stop and remove all data (including volumes):

```
docker-compose down -v
```

## Advanced Configuration

For advanced configuration options, refer to the documentation in the `docs` directory.

---

Thank you for using the Model Distillation UI! We hope it helps you in your knowledge distillation journey.
