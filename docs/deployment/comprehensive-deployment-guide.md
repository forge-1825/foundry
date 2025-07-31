# Comprehensive Deployment Guide for Foundry AI System

## Table of Contents
1. [Overview](#overview)
2. [Deployment Architectures](#deployment-architectures)
3. [Local Development Setup](#local-development-setup)
4. [Docker Deployment](#docker-deployment)
5. [Remote Model Configuration](#remote-model-configuration)
6. [Production Deployment](#production-deployment)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)

## Overview

The Foundry AI system supports multiple deployment configurations to accommodate different use cases:

- **Local Development**: Single machine with GPU for development and testing
- **Docker Deployment**: Containerized deployment for consistency and scalability
- **Remote Models**: SSH-forwarded models from high-performance GPU servers
- **Hybrid Deployment**: Combination of local and remote models

## Deployment Architectures

### 1. Local Development Architecture
```
┌─────────────────────────────────────────┐
│         Local Machine (GPU)              │
├─────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    │
│  │   Frontend  │    │   Backend   │    │
│  │  (React)    │◄──►│  (Flask)    │    │
│  └─────────────┘    └──────┬──────┘    │
│                            │            │
│  ┌─────────────────────────▼──────────┐ │
│  │         Model Servers (vLLM)       │ │
│  ├────────────────┬────────────────┬──┤ │
│  │   Teacher      │    Student     │  │ │
│  │  Port: 8001    │   Port: 8002   │  │ │
│  └────────────────┴────────────────┴──┘ │
└─────────────────────────────────────────┘
```

### 2. Docker Deployment Architecture
```
┌─────────────────────────────────────────┐
│           Docker Host (GPU)              │
├─────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    │
│  │  Frontend   │    │   Backend   │    │
│  │  Container  │◄──►│  Container  │    │
│  └─────────────┘    └──────┬──────┘    │
│                            │            │
│  ┌─────────────────────────▼──────────┐ │
│  │    Model Containers (vLLM)         │ │
│  ├────────────────┬────────────────┬──┤ │
│  │   Teacher      │    Student     │  │ │
│  │  Container     │   Container    │  │ │
│  └────────────────┴────────────────┴──┘ │
└─────────────────────────────────────────┘
```

### 3. Remote Model Architecture
```
┌─────────────────────┐     ┌─────────────────────┐
│   Local Machine     │     │  GPU Server (Remote) │
├─────────────────────┤     ├─────────────────────┤
│  ┌─────────────┐   │     │  ┌─────────────┐    │
│  │  Frontend   │   │     │  │   Teacher   │    │
│  │  + Backend  │   │     │  │   Model     │    │
│  └──────┬──────┘   │     │  └─────────────┘    │
│         │          │     │  ┌─────────────┐    │
│         └──────SSH─┼─────┼─►│   Student   │    │
│                    │     │  │   Model     │    │
│                    │     │  └─────────────┘    │
└─────────────────────┘     └─────────────────────┘
```

## Local Development Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Docker and Docker Compose
- Node.js 16+ and npm

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/foundry.git
cd foundry
```

2. **Set up Python environment**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt
cd ..
```

3. **Set up frontend**
```bash
cd frontend
npm install
cd ..
```

4. **Configure environment variables**
```bash
# Copy example environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Edit environment files with your configuration
# backend/.env
FLASK_ENV=development
USE_REMOTE_MODELS=False
GPU_MEMORY_FRACTION=0.8

# frontend/.env
REACT_APP_API_URL=http://localhost:5000
```

5. **Start model servers**
```bash
# Start teacher model
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-2 \
    --port 8001 \
    --gpu-memory-utilization 0.8 &

# Start student model
python -m vllm.entrypoints.openai.api_server \
    --model microsoft/phi-2 \
    --port 8002 \
    --gpu-memory-utilization 0.8 &
```

6. **Start the application**
```bash
# Terminal 1: Start backend
cd backend
python app/main.py

# Terminal 2: Start frontend
cd frontend
npm start
```

## Docker Deployment

### 1. Basic Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. GPU-Enabled Docker Deployment

```bash
# Use GPU-specific compose file
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up

# Verify GPU access
docker exec foundry-backend nvidia-smi
```

### 3. Custom Model Configuration

Create a `models.yaml` configuration:

```yaml
models:
  teacher:
    model_id: "microsoft/phi-2"
    deployment_type: "local_docker"
    port: 8001
    gpu_memory: 80
    max_batch_size: 32
    max_sequence_length: 2048
    environment:
      CUDA_VISIBLE_DEVICES: "0"

  student:
    model_id: "microsoft/phi-2"
    deployment_type: "local_docker"
    port: 8002
    gpu_memory: 80
    max_batch_size: 32
    max_sequence_length: 2048
    environment:
      CUDA_VISIBLE_DEVICES: "1"
```

Mount the configuration:

```bash
docker-compose up -d \
  -v $(pwd)/models.yaml:/app/config/models.yaml
```

## Remote Model Configuration

### 1. SSH Tunnel Setup

```bash
# Create SSH tunnel to remote GPU server
ssh -N -L 8001:localhost:8001 user@gpu-server.example.com

# Multiple tunnels
ssh -N \
  -L 8001:localhost:8001 \
  -L 8002:localhost:8002 \
  user@gpu-server.example.com
```

### 2. Automatic SSH Tunnel Management

Configure in `models.yaml`:

```yaml
models:
  remote_teacher:
    model_id: "meta-llama/Llama-2-7b-hf"
    deployment_type: "remote_ssh"
    port: 8003
    environment:
      REMOTE_HOST: "gpu-server.example.com"
      REMOTE_PORT: "8000"
      SSH_USER: "user"
      SSH_KEY_PATH: "/path/to/ssh/key"
```

### 3. Environment Configuration

```bash
# Enable remote models
export USE_REMOTE_MODELS=True

# Configure SSH
export SSH_CONFIG_FILE=~/.ssh/config
```

## Production Deployment

### 1. System Requirements

- **CPU**: 16+ cores recommended
- **RAM**: 64GB+ for large models
- **GPU**: NVIDIA A100/H100 for production workloads
- **Storage**: 500GB+ SSD for model storage
- **Network**: 10Gbps+ for distributed deployment

### 2. Production Configuration

Create `production.env`:

```bash
# Flask Configuration
FLASK_ENV=production
SECRET_KEY=<generate-secure-key>
DEBUG=False

# Model Configuration
USE_REMOTE_MODELS=True
MODEL_CACHE_DIR=/data/models
MAX_CONCURRENT_REQUESTS=100

# Security
ENABLE_CORS=True
ALLOWED_ORIGINS=https://your-domain.com
API_KEY_REQUIRED=True

# Monitoring
ENABLE_METRICS=True
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### 3. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: foundry-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: foundry-backend
  template:
    metadata:
      labels:
        app: foundry-backend
    spec:
      containers:
      - name: backend
        image: foundry/backend:latest
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: foundry-backend-service
spec:
  selector:
    app: foundry-backend
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

### 4. Load Balancing

Using NGINX:

```nginx
upstream foundry_backend {
    least_conn;
    server backend1:5000 weight=1;
    server backend2:5000 weight=1;
    server backend3:5000 weight=1;
}

server {
    listen 80;
    server_name foundry.example.com;

    location / {
        proxy_pass http://foundry_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring and Maintenance

### 1. Health Checks

```bash
# Check system health
curl http://localhost:5000/health

# Check model availability
curl http://localhost:5000/api/models/status

# Check pipeline status
curl http://localhost:5000/api/pipeline/status
```

### 2. Monitoring Setup

Using Prometheus and Grafana:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'foundry'
    static_configs:
      - targets: ['localhost:9090']
```

### 3. Log Management

```bash
# Aggregate logs
docker-compose logs -f > foundry.log

# Rotate logs
logrotate -f /etc/logrotate.d/foundry

# Send to centralized logging
fluentd -c /etc/fluent/fluent.conf
```

### 4. Backup Strategy

```bash
# Backup models
rsync -avz /data/models/ /backup/models/

# Backup configuration
tar -czf config-backup.tar.gz config/

# Backup database (if applicable)
pg_dump foundry > foundry-backup.sql
```

## Troubleshooting

### Common Issues and Solutions

#### 1. GPU Out of Memory
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size in configuration
gpu_memory: 60  # Reduce from 80

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Model Loading Failures
```bash
# Check model path
ls -la /data/models/

# Verify model access
python -c "from transformers import AutoModel; AutoModel.from_pretrained('model-name')"

# Check disk space
df -h
```

#### 3. Connection Issues
```bash
# Test connectivity
curl -v http://localhost:8001/v1/models

# Check firewall rules
sudo iptables -L

# Verify SSH tunnel
ssh -v -N -L 8001:localhost:8001 user@remote
```

#### 4. Performance Issues
```bash
# Profile CPU usage
htop

# Check memory usage
free -h

# Monitor disk I/O
iotop

# Check network latency
ping remote-server
```

### Debug Mode

Enable debug logging:

```python
# In backend/.env
DEBUG=True
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart
```

### Support Resources

- **Documentation**: https://foundry-docs.example.com
- **Issue Tracker**: https://github.com/your-org/foundry/issues
- **Community Forum**: https://forum.foundry.ai
- **Email Support**: support@foundry.ai

## Best Practices

1. **Security**
   - Use HTTPS in production
   - Implement API key authentication
   - Regular security updates
   - Network isolation for model servers

2. **Performance**
   - Use model caching
   - Implement request queuing
   - Monitor resource usage
   - Scale horizontally when needed

3. **Reliability**
   - Implement health checks
   - Use process managers (systemd/supervisor)
   - Set up automated backups
   - Plan for disaster recovery

4. **Maintenance**
   - Regular model updates
   - Log rotation
   - Performance tuning
   - Documentation updates