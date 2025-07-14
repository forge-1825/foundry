#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if we should use remote models
if [ "$USE_REMOTE_MODELS" = "true" ] || [ "$USE_REMOTE_MODELS" = "True" ]; then
    echo "Starting with remote models configuration (no local GPU)..."
    docker-compose up -d
else
    if [ "$ENABLE_GPU" = "1" ]; then
        echo "Starting with local GPU support..."
        docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
    else
        echo "Starting with local CPU configuration..."
        docker-compose up -d
    fi
fi

echo "Services started. Check status with: docker-compose ps"