from fastapi import APIRouter, HTTPException
import subprocess
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create router
docker_bp = APIRouter(prefix="/api/docker", tags=["docker"])

@docker_bp.get("/containers")
async def get_docker_containers():
    """Get all running Docker containers"""
    try:
        # Run docker ps command and get output in JSON format
        result = subprocess.run(
            ["docker", "ps", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    container = json.loads(line)
                    # Parse ports
                    ports = []
                    if container.get('Ports'):
                        port_strings = container['Ports'].split(', ')
                        for port_str in port_strings:
                            if '->' in port_str:
                                public_port, private_port = port_str.split('->')[0], port_str.split('->')[1]
                                public_port = public_port.strip()
                                private_port, port_type = private_port.split('/')[0].strip(), private_port.split('/')[1].strip()
                                ports.append({
                                    'PublicPort': int(public_port),
                                    'PrivatePort': int(private_port),
                                    'Type': port_type
                                })
                    container['Ports'] = ports
                    containers.append(container)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing container JSON: {e}")
        
        return containers
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running docker ps: {e}")
        raise HTTPException(status_code=500, detail=f"Error running docker ps: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@docker_bp.get("/ps")
async def get_docker_ps():
    """Get raw output from docker ps command"""
    try:
        # Run docker ps command with all information
        result = subprocess.run(
            ["docker", "ps", "--all", "--no-trunc"],
            capture_output=True,
            text=True,
            check=True
        )
        
        return {"output": result.stdout}
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running docker ps: {e}")
        raise HTTPException(status_code=500, detail=f"Error running docker ps: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@docker_bp.get("/images")
async def get_docker_images():
    """Get all Docker images"""
    try:
        # Run docker images command and get output in JSON format
        result = subprocess.run(
            ["docker", "images", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        images = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    image = json.loads(line)
                    images.append(image)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing image JSON: {e}")
        
        return images
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running docker images: {e}")
        raise HTTPException(status_code=500, detail=f"Error running docker images: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@docker_bp.get("/info")
async def get_docker_info():
    """Get Docker system information"""
    try:
        # Run docker info command
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        info = json.loads(result.stdout)
        return info
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running docker info: {e}")
        raise HTTPException(status_code=500, detail=f"Error running docker info: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing docker info JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing docker info JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
