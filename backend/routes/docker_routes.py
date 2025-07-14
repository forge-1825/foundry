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
                    
                    # Extract container name (remove leading slash if present)
                    container_name = container.get('Names', '')
                    if container_name.startswith('/'):
                        container_name = container_name[1:]
                    
                    # Parse ports
                    ports = []
                    port_number = None
                    if container.get('Ports'):
                        port_strings = container['Ports'].split(', ')
                        for port_str in port_strings:
                            if '->' in port_str:
                                public_part, private_part = port_str.split('->')
                                public_part = public_part.strip()
                                
                                # Extract port number from public part (handles IP:port format)
                                if ':' in public_part:
                                    public_port = public_part.split(':')[-1]
                                else:
                                    public_port = public_part
                                
                                # Extract private port and type
                                private_port, port_type = private_part.split('/')[0].strip(), private_part.split('/')[1].strip()
                                
                                try:
                                    ports.append({
                                        'PublicPort': int(public_port),
                                        'PrivatePort': int(private_port),
                                        'Type': port_type
                                    })
                                    # Store the first public port for easy access
                                    if port_number is None:
                                        port_number = int(public_port)
                                except ValueError as e:
                                    logger.warning(f"Could not parse port: {port_str} - {e}")
                    
                    container['Ports'] = ports
                    container['Name'] = container_name
                    container['Port'] = port_number
                    
                    # Determine container type based on port or name
                    container_type = "Unknown"
                    if port_number == 8000:
                        container_type = "Teacher Model"
                    elif port_number == 8001:
                        container_type = "Student Model"
                    elif port_number == 8002:
                        container_type = "Student Model"
                    elif port_number == 8003:
                        container_type = "Distilled Model"
                    elif 'teacher' in container_name.lower():
                        container_type = "Teacher Model"
                    elif 'student' in container_name.lower():
                        container_type = "Student Model"
                    elif 'distilled' in container_name.lower():
                        container_type = "Distilled Model"
                    
                    container['Type'] = container_type
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
