# Model Setup Guide

This guide provides comprehensive instructions for setting up model containers for the Foundry pipeline.

## Overview

The Foundry pipeline requires several Large Language Models (LLMs) to function:
- **Teacher Model**: Llama 3 8B Instruct (for generating high-quality training data)
- **Student Model**: Phi-3 Mini 4k Instruct (the model being trained)
- **Distilled Model**: The fine-tuned version of the student model

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (for GPU acceleration)
- At least 16GB RAM (32GB recommended)
- At least 50GB free disk space

## Setup Methods

### Method 1: Docker Compose (Recommended)

1. **GPU Setup**:
   ```bash
   cd /home/tachyon/Foundry_entry/Foundry/foundry
   docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
   ```

2. **CPU Setup** (fallback):
   ```bash
   cd /home/tachyon/Foundry_entry/Foundry/foundry
   docker-compose up -d
   ```

3. **Verify containers are running**:
   ```bash
   docker ps
   ```

### Method 2: Remote Models via SSH

If models are hosted on a remote server:

1. **Set environment variable**:
   ```bash
   export USE_REMOTE_MODELS=true
   ```

2. **Configure SSH tunnels**:
   ```bash
   # Teacher model
   ssh -L 8000:localhost:8000 user@remote-server -N &
   
   # Student model  
   ssh -L 8002:localhost:8002 user@remote-server -N &
   
   # Distilled model (if available)
   ssh -L 8003:localhost:8003 user@remote-server -N &
   ```

3. **Use SSH-aware scripts**:
   - Use `teacher_pair_generation_vllm_ssh.py` for teacher pair generation
   - The vLLM client will automatically detect remote models

### Method 3: Manual vLLM Server Setup

For custom deployments:

1. **Install vLLM**:
   ```bash
   pip install vllm
   ```

2. **Start Teacher Model**:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Meta-Llama-3-8B-Instruct \
     --port 8000 \
     --dtype auto \
     --api-key token-abc123
   ```

3. **Start Student Model**:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model microsoft/Phi-3-mini-4k-instruct \
     --port 8002 \
     --dtype auto \
     --api-key token-abc123
   ```

## Model Configuration

### Default Ports
- Teacher Model: 8000
- Student Model: 8002  
- Distilled Model: 8003

### Environment Variables
- `USE_REMOTE_MODELS`: Set to "true" for remote model access
- `VLLM_API_KEY`: API key for vLLM servers (default: "token-abc123")
- `CUDA_VISIBLE_DEVICES`: GPU selection (e.g., "0,1" for GPUs 0 and 1)

### Model Requirements

#### Teacher Model (Llama 3 8B)
- **RAM**: 16GB minimum
- **VRAM**: 16GB for full precision, 8GB with quantization
- **Disk**: ~15GB for model weights

#### Student Model (Phi-3 Mini)
- **RAM**: 8GB minimum
- **VRAM**: 4GB
- **Disk**: ~8GB for model weights

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Enable quantization: Add `--quantization awq` to vLLM command
   - Reduce batch size in pipeline configuration
   - Use CPU offloading

2. **Connection Refused**:
   - Check if containers are running: `docker ps`
   - Verify ports are not blocked by firewall
   - Check SSH tunnels are active (for remote setup)

3. **Model Not Found**:
   - Ensure HuggingFace token is set for gated models
   - Check internet connectivity for model downloads
   - Verify model names are correct

4. **CUDA Errors**:
   - Run `nvidia-smi` to check GPU availability
   - Ensure NVIDIA Container Toolkit is installed
   - Check CUDA version compatibility

### Verification Steps

1. **Check Model Availability**:
   ```bash
   curl http://localhost:8000/v1/models
   curl http://localhost:8002/v1/models
   ```

2. **Test Model Response**:
   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "meta-llama/Meta-Llama-3-8B-Instruct",
       "prompt": "Hello, how are you?",
       "max_tokens": 50
     }'
   ```

3. **Monitor Resources**:
   ```bash
   # GPU usage
   nvidia-smi -l 1
   
   # Docker container resources
   docker stats
   ```

## Performance Optimization

### GPU Optimization
- Use `--gpu-memory-utilization 0.9` for maximum GPU usage
- Enable `--enable-prefix-caching` for repeated prompts
- Use `--max-model-len` to limit context length

### CPU Optimization
- Increase `--cpu-offload-gb` for models larger than VRAM
- Use `--num-cpu-blocks-override` for better CPU cache usage

### Network Optimization (Remote Models)
- Use compression in SSH: `ssh -C`
- Consider VPN for better stability
- Monitor latency with `ping` to remote server

## Security Considerations

1. **API Keys**: Change default API keys in production
2. **Network**: Use TLS/SSL for remote connections
3. **Firewall**: Restrict ports to trusted IPs only
4. **Container Security**: Run containers with limited privileges

## Quick Start Checklist

- [ ] Docker/Docker Compose installed
- [ ] GPU drivers and CUDA installed (for GPU)
- [ ] Sufficient RAM and disk space
- [ ] Network connectivity for model downloads
- [ ] Ports 8000, 8002, 8003 available
- [ ] Environment variables configured
- [ ] Containers started and verified
- [ ] Test queries successful

## Support

For issues specific to:
- **vLLM**: Check [vLLM documentation](https://docs.vllm.ai)
- **Docker**: See Docker logs: `docker logs <container-name>`
- **Models**: Verify on [HuggingFace](https://huggingface.co)
- **Pipeline**: Check `backend/logs/` for detailed logs