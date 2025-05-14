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

class VLLMClient:
    """Client for interacting with vLLM servers using OpenAI-compatible API"""
    
    def __init__(self, base_url: str = "http://localhost:8002/v1", api_key: str = "dummy-key"):
        """
        Initialize the vLLM client.
        
        Args:
            base_url: Base URL for the vLLM server
            api_key: API key (not used by vLLM, but required for OpenAI compatibility)
        """
        self.base_url = base_url
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
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying model: {e}")
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
    base_url = f"http://localhost:{port}/v1"
    return VLLMClient(base_url=base_url)

def get_docker_container_ports() -> Dict[str, int]:
    """
    Get the ports for Docker containers running vLLM servers.
    
    Returns:
        Dictionary mapping container names to ports
    """
    import subprocess
    
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
            
            # Look for port mappings like 0.0.0.0:8002->80/tcp
            if '8000->80' in ports_str:
                container_ports[name] = 8000
            elif '8001->80' in ports_str:
                container_ports[name] = 8001
            elif '8002->80' in ports_str:
                container_ports[name] = 8002
            elif '->80' in ports_str:
                # Extract the port number
                import re
                match = re.search(r':(\d+)->80', ports_str)
                if match:
                    container_ports[name] = int(match.group(1))
        
        return container_ports
    except Exception as e:
        logger.error(f"Error getting Docker container ports: {e}")
        return {}

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
