"""
vLLM Client for querying models with OpenAI-compatible API
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect if running in Docker container
def is_running_in_docker():
    """Check if the code is running inside a Docker container"""
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == '1'

# Get the appropriate host for accessing services from container
def get_host():
    """Get the appropriate hostname for accessing host services"""
    # Check if we should use remote models (SSH forwarded) or local Docker models
    use_remote_models = os.environ.get('USE_REMOTE_MODELS', 'True').lower() == 'true'
    
    if use_remote_models and is_running_in_docker():
        # Remote models accessed via SSH port forwarding from host
        return 'host.docker.internal'
    elif use_remote_models:
        # Remote models accessed directly (not in container)
        return 'localhost'
    else:
        # Local Docker models - use localhost even from container
        # since they would be in the same Docker network
        return 'localhost'

# Default host for vLLM connections
VLLM_HOST = get_host()
logger.info(f"Using VLLM_HOST: {VLLM_HOST} (USE_REMOTE_MODELS={os.environ.get('USE_REMOTE_MODELS', 'True')})")

class VLLMClient:
    """Client for interacting with vLLM servers using OpenAI-compatible API"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """
        Initialize the vLLM client.
        
        Args:
            base_url: Base URL for the vLLM server
            api_key: API key (optional, defaults to environment variable VLLM_API_KEY or 'not-needed')
        """
        if base_url is None:
            base_url = f"http://{VLLM_HOST}:8002/v1"
        self.base_url = base_url
        
        # Use provided api_key, or fall back to environment variable, or default
        if api_key is None:
            api_key = os.environ.get('VLLM_API_KEY', 'not-needed')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def query_model(self, prompt: str, model: str = "/model", max_tokens: int = 500, 
                   temperature: float = 0.7) -> Dict[str, Any]:
        """
        Query the model with a prompt.
        
        Args:
            prompt: The prompt to send to the model
            model: The model ID to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            
        Returns:
            The model's response
        """
        url = f"{self.base_url}/completions"
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to model server at {self.base_url}")
            logger.error("Please ensure:")
            logger.error("1. Model containers are running (docker ps)")
            logger.error("2. Ports are accessible (check firewall)")
            logger.error("3. SSH tunnels are active (if using remote models)")
            raise ConnectionError(f"Model server unavailable at {self.base_url}")
        except requests.exceptions.Timeout:
            logger.error("Request timed out - model may be overloaded or unresponsive")
            raise TimeoutError("Model request timed out after 60 seconds")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Model '{model}' not found on server")
                logger.error("Use get_available_models() to see available models")
            elif e.response.status_code == 401:
                logger.error("Authentication failed - check API key")
            else:
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Unexpected error querying model: {e}")
            raise
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Get available models from the vLLM server.
        
        Returns:
            List of available models
        """
        url = f"{self.base_url}/models"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.ConnectionError:
            logger.warning(f"Cannot connect to {self.base_url} - server may not be running")
            return []
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout connecting to {self.base_url}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting available models: {e}")
            return []

def get_vllm_client_for_port(port: int) -> VLLMClient:
    """
    Get a vLLM client for a specific port.
    
    Args:
        port: The port number
        
    Returns:
        A vLLM client
    """
    base_url = f"http://{VLLM_HOST}:{port}/v1"
    return VLLMClient(base_url=base_url)

def get_docker_container_ports() -> Dict[str, int]:
    """
    Get the ports for Docker containers running vLLM servers.
    
    Returns:
        Dictionary mapping container names to ports
    """
    import subprocess
    import re
    
    try:
        # Get list of running containers with port mappings
        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}\t{{.Ports}}'],
            capture_output=True, text=True, check=True
        )
        
        container_ports = {}
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                continue
                
            name, ports_str = parts
            
            # More flexible port mapping detection
            # Look for patterns like:
            # - 0.0.0.0:8000->8000/tcp
            # - 0.0.0.0:8001->80/tcp
            # - :::8002->8000/tcp
            # - 8003/tcp
            
            # Try to find any port mapping
            port_patterns = [
                r':(\d+)->8000',  # Maps to vLLM default port
                r':(\d+)->80',     # Maps to HTTP port
                r':(\d+)->(\d+)',  # Any port mapping
                r'^(\d+)/tcp',     # Direct port exposure
            ]
            
            for pattern in port_patterns:
                match = re.search(pattern, ports_str)
                if match:
                    port = int(match.group(1))
                    # Only consider typical vLLM ports or ports in common range
                    if 8000 <= port <= 8100 or port in [80, 443]:
                        container_ports[name] = port
                        logger.info(f"Found container {name} on port {port}")
                        break
        
        return container_ports
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker command failed: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error getting Docker container ports: {e}")
        return {}

def detect_all_vllm_servers(check_ssh_ports: bool = True) -> List[Dict[str, any]]:
    """
    Detect all available vLLM servers including Docker containers and SSH-forwarded ports.
    
    Args:
        check_ssh_ports: Whether to check common SSH-forwarded ports
        
    Returns:
        List of available servers with their details
    """
    servers = []
    
    # Get Docker container ports
    container_ports = get_docker_container_ports()
    for name, port in container_ports.items():
        servers.append({
            'name': name,
            'port': port,
            'type': 'docker',
            'source_type': 'docker',  # Keep track of original source
            'status': 'unknown'
        })
    
    # Check common SSH-forwarded ports if requested
    if check_ssh_ports:
        ssh_ports = [8000, 8001, 8002, 8003, 8004, 8005]
        for port in ssh_ports:
            # Skip if already detected via Docker
            if port not in container_ports.values():
                # Test if port is responding
                try:
                    client = VLLMClient(base_url=f"http://{VLLM_HOST}:{port}/v1")
                    models = client.get_available_models()
                    if models:
                        servers.append({
                            'name': f'ssh_forwarded_port_{port}',
                            'port': port,
                            'type': 'ssh',
                            'source_type': 'ssh',  # Keep track of original source
                            'status': 'active',
                            'models': models
                        })
                except Exception:
                    # Port not responding or not a vLLM server
                    pass
    
    # Test each server's status
    for server in servers:
        if server['status'] == 'unknown':
            try:
                client = VLLMClient(base_url=f"http://{VLLM_HOST}:{server['port']}/v1")
                models = client.get_available_models()
                server['status'] = 'active' if models else 'inactive'
                server['models'] = models
            except Exception:
                server['status'] = 'inactive'
    
    return servers

def query_vllm_model(port: int, prompt: str, model: str = "/model", 
                    max_tokens: int = 500, temperature: float = 0.7) -> str:
    """
    Query a vLLM model with a prompt.
    
    Args:
        port: The port number
        prompt: The prompt to send to the model
        model: The model ID to use
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        
    Returns:
        The model's response text
    """
    client = get_vllm_client_for_port(port)
    
    try:
        response = client.query_model(prompt, model, max_tokens, temperature)
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["text"]
        else:
            return "No response from model"
    except Exception as e:
        logger.error(f"Error querying model: {e}")
        return f"Error: {str(e)}"
