"""
Model Deployment Manager for Foundry
Handles both local Docker and remote SSH-forwarded model deployments
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import requests
import yaml
import docker
from datetime import datetime, timedelta

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.unified_error_handler import (
    FoundryError, ModelError, NetworkError, ConfigurationError,
    handle_errors, with_retry, error_context, get_error_logger
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentType(Enum):
    """Model deployment types"""
    LOCAL_DOCKER = "local_docker"
    REMOTE_SSH = "remote_ssh"
    KUBERNETES = "kubernetes"
    CLOUD = "cloud"


class ModelStatus(Enum):
    """Model server status"""
    UNKNOWN = "unknown"
    STARTING = "starting"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    model_id: str
    deployment_type: DeploymentType
    port: int
    gpu_memory: Optional[int] = None
    max_batch_size: Optional[int] = 32
    max_sequence_length: Optional[int] = 2048
    environment: Dict[str, str] = field(default_factory=dict)
    health_check_endpoint: str = "/v1/models"
    startup_timeout: int = 300  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "deployment_type": self.deployment_type.value,
            "port": self.port,
            "gpu_memory": self.gpu_memory,
            "max_batch_size": self.max_batch_size,
            "max_sequence_length": self.max_sequence_length,
            "environment": self.environment,
            "health_check_endpoint": self.health_check_endpoint,
            "startup_timeout": self.startup_timeout
        }


@dataclass
class ModelDeployment:
    """Model deployment information"""
    config: ModelConfig
    status: ModelStatus = ModelStatus.UNKNOWN
    container_id: Optional[str] = None
    start_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if deployment is healthy"""
        return self.status == ModelStatus.READY
    
    def uptime(self) -> Optional[timedelta]:
        """Get deployment uptime"""
        if self.start_time:
            return datetime.now() - self.start_time
        return None


class HealthChecker:
    """Health check manager for model deployments"""
    
    def __init__(self, check_interval: int = 30, max_failures: int = 3):
        self.check_interval = check_interval
        self.max_failures = max_failures
        self.running = False
        self.thread = None
        self.deployments: Dict[str, ModelDeployment] = {}
        
    def add_deployment(self, deployment: ModelDeployment):
        """Add deployment to health monitoring"""
        self.deployments[deployment.config.name] = deployment
        
    def remove_deployment(self, name: str):
        """Remove deployment from health monitoring"""
        self.deployments.pop(name, None)
        
    def start(self):
        """Start health checking thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._health_check_loop, daemon=True)
            self.thread.start()
            
    def stop(self):
        """Stop health checking"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
            
    def _health_check_loop(self):
        """Main health check loop"""
        while self.running:
            for deployment in list(self.deployments.values()):
                try:
                    self._check_deployment_health(deployment)
                except Exception as e:
                    logger.error(f"Health check failed for {deployment.config.name}: {e}")
            time.sleep(self.check_interval)
            
    @with_retry(max_retries=2, backoff_factor=1.5)
    def _check_deployment_health(self, deployment: ModelDeployment):
        """Check health of a single deployment"""
        config = deployment.config
        
        # Determine host based on deployment type
        if config.deployment_type == DeploymentType.LOCAL_DOCKER:
            host = "localhost"
        elif config.deployment_type == DeploymentType.REMOTE_SSH:
            # Use host.docker.internal if running in Docker
            host = "host.docker.internal" if self._is_in_docker() else "localhost"
        else:
            host = "localhost"
            
        url = f"http://{host}:{config.port}{config.health_check_endpoint}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Update deployment status
            deployment.status = ModelStatus.READY
            deployment.last_health_check = datetime.now()
            deployment.health_check_failures = 0
            
            # Update metrics
            if "data" in data:
                deployment.metrics["available_models"] = len(data.get("data", []))
                deployment.metrics["model_names"] = [m.get("id") for m in data.get("data", [])]
                
        except requests.exceptions.RequestException as e:
            deployment.health_check_failures += 1
            
            if deployment.health_check_failures >= self.max_failures:
                deployment.status = ModelStatus.UNHEALTHY
                logger.error(f"Model {config.name} marked as unhealthy after {self.max_failures} failures")
            else:
                logger.warning(f"Health check failed for {config.name}: {e}")
                
    def _is_in_docker(self) -> bool:
        """Check if running inside Docker"""
        return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == '1'


class DockerModelManager:
    """Manages Docker-based model deployments"""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            raise ConfigurationError(f"Docker not available: {e}")
            
    @handle_errors(category=ErrorCategory.MODEL)
    def deploy_model(self, config: ModelConfig) -> ModelDeployment:
        """Deploy a model using Docker"""
        deployment = ModelDeployment(config=config)
        
        try:
            # Build Docker command
            docker_cmd = self._build_docker_command(config)
            
            # Check if container already exists
            existing = self._find_existing_container(config.name)
            if existing:
                logger.info(f"Stopping existing container for {config.name}")
                existing.stop()
                existing.remove()
                
            # Run container
            logger.info(f"Starting Docker container for {config.name}")
            container = self.client.containers.run(
                **docker_cmd,
                detach=True,
                name=f"foundry-model-{config.name}"
            )
            
            deployment.container_id = container.id
            deployment.status = ModelStatus.STARTING
            deployment.start_time = datetime.now()
            
            # Wait for model to be ready
            if self._wait_for_ready(deployment):
                deployment.status = ModelStatus.READY
                logger.info(f"Model {config.name} is ready on port {config.port}")
            else:
                deployment.status = ModelStatus.ERROR
                raise ModelError(f"Model {config.name} failed to start", config.name)
                
        except docker.errors.APIError as e:
            deployment.status = ModelStatus.ERROR
            raise ModelError(f"Docker API error: {e}", config.name)
            
        return deployment
        
    def _build_docker_command(self, config: ModelConfig) -> Dict[str, Any]:
        """Build Docker run command"""
        cmd = {
            "image": "vllm/vllm-openai:latest",
            "command": [
                "--model", config.model_id,
                "--port", "8000",  # Internal port
                "--max-model-len", str(config.max_sequence_length),
            ],
            "ports": {
                "8000/tcp": config.port  # Map to external port
            },
            "environment": config.environment
        }
        
        # Add GPU support if available
        if config.gpu_memory:
            cmd["device_requests"] = [
                docker.types.DeviceRequest(
                    device_ids=["0"],  # Use first GPU
                    capabilities=[["gpu"]]
                )
            ]
            cmd["command"].extend([
                "--gpu-memory-utilization", str(config.gpu_memory / 100)
            ])
            
        return cmd
        
    def _find_existing_container(self, name: str) -> Optional[Any]:
        """Find existing container by name"""
        try:
            return self.client.containers.get(f"foundry-model-{name}")
        except docker.errors.NotFound:
            return None
            
    def _wait_for_ready(self, deployment: ModelDeployment, timeout: Optional[int] = None) -> bool:
        """Wait for model to be ready"""
        timeout = timeout or deployment.config.startup_timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check container status
                container = self.client.containers.get(deployment.container_id)
                if container.status != "running":
                    logger.error(f"Container {deployment.config.name} is not running: {container.status}")
                    return False
                    
                # Check health endpoint
                url = f"http://localhost:{deployment.config.port}{deployment.config.health_check_endpoint}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    return True
                    
            except (requests.exceptions.RequestException, docker.errors.NotFound):
                pass
                
            time.sleep(5)
            
        return False
        
    def stop_model(self, name: str):
        """Stop a model deployment"""
        container = self._find_existing_container(name)
        if container:
            container.stop()
            container.remove()
            logger.info(f"Stopped model {name}")
            
    def get_container_logs(self, name: str, lines: int = 100) -> str:
        """Get container logs"""
        container = self._find_existing_container(name)
        if container:
            return container.logs(tail=lines).decode('utf-8')
        return ""


class SSHModelManager:
    """Manages SSH-forwarded model deployments"""
    
    def __init__(self, ssh_config: Optional[Dict[str, Any]] = None):
        self.ssh_config = ssh_config or {}
        self.tunnels: Dict[str, subprocess.Popen] = {}
        
    @handle_errors(category=ErrorCategory.NETWORK)
    def create_tunnel(self, config: ModelConfig, remote_host: str, remote_port: int) -> ModelDeployment:
        """Create SSH tunnel for remote model"""
        deployment = ModelDeployment(config=config)
        
        # Build SSH command
        ssh_cmd = [
            "ssh",
            "-N",  # No command execution
            "-L", f"{config.port}:localhost:{remote_port}",  # Port forwarding
        ]
        
        # Add SSH options
        if "user" in self.ssh_config:
            ssh_cmd.extend([f"{self.ssh_config['user']}@{remote_host}"])
        else:
            ssh_cmd.append(remote_host)
            
        if "identity_file" in self.ssh_config:
            ssh_cmd.extend(["-i", self.ssh_config["identity_file"]])
            
        # Additional options
        ssh_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3"
        ])
        
        try:
            # Start SSH tunnel
            logger.info(f"Creating SSH tunnel for {config.name} on port {config.port}")
            process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.tunnels[config.name] = process
            deployment.status = ModelStatus.STARTING
            deployment.start_time = datetime.now()
            
            # Wait for tunnel to be established
            time.sleep(2)
            
            # Check if tunnel is working
            if self._check_tunnel(config):
                deployment.status = ModelStatus.READY
                logger.info(f"SSH tunnel for {config.name} established on port {config.port}")
            else:
                deployment.status = ModelStatus.ERROR
                raise NetworkError(f"SSH tunnel failed for {config.name}", remote_host)
                
        except subprocess.SubprocessError as e:
            deployment.status = ModelStatus.ERROR
            raise NetworkError(f"SSH command failed: {e}", remote_host)
            
        return deployment
        
    def _check_tunnel(self, config: ModelConfig) -> bool:
        """Check if SSH tunnel is working"""
        try:
            url = f"http://localhost:{config.port}{config.health_check_endpoint}"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
            
    def stop_tunnel(self, name: str):
        """Stop SSH tunnel"""
        if name in self.tunnels:
            process = self.tunnels[name]
            process.terminate()
            process.wait(timeout=5)
            del self.tunnels[name]
            logger.info(f"Stopped SSH tunnel for {name}")


class ModelDeploymentManager:
    """Main model deployment manager"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("config/models.yaml")
        self.deployments: Dict[str, ModelDeployment] = {}
        self.docker_manager = DockerModelManager()
        self.ssh_manager = SSHModelManager()
        self.health_checker = HealthChecker()
        self.health_checker.start()
        
        # Load configuration
        self.load_configuration()
        
    def load_configuration(self):
        """Load model deployment configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                self.model_configs = self._parse_configs(config)
        else:
            self.model_configs = {}
            
    def _parse_configs(self, config: Dict[str, Any]) -> Dict[str, ModelConfig]:
        """Parse model configurations"""
        configs = {}
        
        for name, model_config in config.get("models", {}).items():
            deployment_type = DeploymentType(model_config.get("deployment_type", "local_docker"))
            
            configs[name] = ModelConfig(
                name=name,
                model_id=model_config["model_id"],
                deployment_type=deployment_type,
                port=model_config["port"],
                gpu_memory=model_config.get("gpu_memory"),
                max_batch_size=model_config.get("max_batch_size", 32),
                max_sequence_length=model_config.get("max_sequence_length", 2048),
                environment=model_config.get("environment", {}),
                health_check_endpoint=model_config.get("health_check_endpoint", "/v1/models"),
                startup_timeout=model_config.get("startup_timeout", 300)
            )
            
        return configs
        
    @handle_errors(category=ErrorCategory.MODEL)
    def deploy_model(self, name: str, config: Optional[ModelConfig] = None) -> ModelDeployment:
        """Deploy a model"""
        # Use provided config or load from configuration
        if config is None:
            if name not in self.model_configs:
                raise ConfigurationError(f"Model {name} not found in configuration")
            config = self.model_configs[name]
            
        # Check if already deployed
        if name in self.deployments and self.deployments[name].is_healthy():
            logger.info(f"Model {name} is already deployed and healthy")
            return self.deployments[name]
            
        # Deploy based on type
        with error_context("model_deployment", {"model": name, "type": config.deployment_type.value}):
            if config.deployment_type == DeploymentType.LOCAL_DOCKER:
                deployment = self.docker_manager.deploy_model(config)
            elif config.deployment_type == DeploymentType.REMOTE_SSH:
                # For SSH, we need remote host info (should be in config)
                remote_host = config.environment.get("REMOTE_HOST", "localhost")
                remote_port = int(config.environment.get("REMOTE_PORT", "8000"))
                deployment = self.ssh_manager.create_tunnel(config, remote_host, remote_port)
            else:
                raise ConfigurationError(f"Unsupported deployment type: {config.deployment_type}")
                
        # Add to deployments and health monitoring
        self.deployments[name] = deployment
        self.health_checker.add_deployment(deployment)
        
        return deployment
        
    def stop_model(self, name: str):
        """Stop a model deployment"""
        if name not in self.deployments:
            logger.warning(f"Model {name} is not deployed")
            return
            
        deployment = self.deployments[name]
        config = deployment.config
        
        # Remove from health monitoring
        self.health_checker.remove_deployment(name)
        
        # Stop based on type
        if config.deployment_type == DeploymentType.LOCAL_DOCKER:
            self.docker_manager.stop_model(name)
        elif config.deployment_type == DeploymentType.REMOTE_SSH:
            self.ssh_manager.stop_tunnel(name)
            
        # Remove from deployments
        del self.deployments[name]
        logger.info(f"Stopped model {name}")
        
    def get_deployment_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        if name not in self.deployments:
            return None
            
        deployment = self.deployments[name]
        return {
            "name": name,
            "status": deployment.status.value,
            "deployment_type": deployment.config.deployment_type.value,
            "port": deployment.config.port,
            "uptime": str(deployment.uptime()) if deployment.uptime() else None,
            "last_health_check": deployment.last_health_check.isoformat() if deployment.last_health_check else None,
            "health_check_failures": deployment.health_check_failures,
            "metrics": deployment.metrics
        }
        
    def get_all_deployments(self) -> Dict[str, Dict[str, Any]]:
        """Get all deployment statuses"""
        return {
            name: self.get_deployment_status(name)
            for name in self.deployments
        }
        
    def validate_deployment(self, name: str) -> Tuple[bool, Optional[str]]:
        """Validate a deployment is working correctly"""
        if name not in self.deployments:
            return False, "Deployment not found"
            
        deployment = self.deployments[name]
        
        # Check status
        if not deployment.is_healthy():
            return False, f"Deployment is {deployment.status.value}"
            
        # Try to query the model
        try:
            config = deployment.config
            url = f"http://localhost:{config.port}/v1/completions"
            
            response = requests.post(
                url,
                json={
                    "model": config.model_id,
                    "prompt": "Hello, ",
                    "max_tokens": 5,
                    "temperature": 0.1
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return True, None
            else:
                return False, f"Model query failed: {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Model query error: {e}"
            
    def cleanup(self):
        """Cleanup all deployments"""
        logger.info("Cleaning up all model deployments")
        
        # Stop health checker
        self.health_checker.stop()
        
        # Stop all deployments
        for name in list(self.deployments.keys()):
            self.stop_model(name)


# Configuration validation
class DeploymentConfigValidator:
    """Validates deployment configurations"""
    
    @staticmethod
    def validate_config(config: ModelConfig) -> List[str]:
        """Validate a model configuration"""
        errors = []
        
        # Check required fields
        if not config.name:
            errors.append("Model name is required")
        if not config.model_id:
            errors.append("Model ID is required")
        if not config.port or config.port < 1024 or config.port > 65535:
            errors.append("Valid port number required (1024-65535)")
            
        # Check deployment-specific requirements
        if config.deployment_type == DeploymentType.LOCAL_DOCKER:
            # Check if Docker is available
            try:
                docker.from_env().ping()
            except docker.errors.DockerException:
                errors.append("Docker is not available for local deployment")
                
        elif config.deployment_type == DeploymentType.REMOTE_SSH:
            # Check SSH configuration
            if "REMOTE_HOST" not in config.environment:
                errors.append("REMOTE_HOST required for SSH deployment")
                
        # Check resource requirements
        if config.gpu_memory and (config.gpu_memory < 0 or config.gpu_memory > 100):
            errors.append("GPU memory must be between 0 and 100")
            
        return errors
        
    @staticmethod
    def validate_config_file(config_file: Path) -> Tuple[bool, List[str]]:
        """Validate a configuration file"""
        errors = []
        
        if not config_file.exists():
            return False, ["Configuration file not found"]
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if not isinstance(config, dict):
                errors.append("Configuration must be a dictionary")
                return False, errors
                
            models = config.get("models", {})
            if not models:
                errors.append("No models defined in configuration")
                
            for name, model_config in models.items():
                # Create ModelConfig and validate
                try:
                    deployment_type = DeploymentType(model_config.get("deployment_type", "local_docker"))
                    mc = ModelConfig(
                        name=name,
                        model_id=model_config.get("model_id", ""),
                        deployment_type=deployment_type,
                        port=model_config.get("port", 0),
                        gpu_memory=model_config.get("gpu_memory"),
                        environment=model_config.get("environment", {})
                    )
                    
                    model_errors = DeploymentConfigValidator.validate_config(mc)
                    if model_errors:
                        errors.extend([f"{name}: {e}" for e in model_errors])
                        
                except (ValueError, KeyError) as e:
                    errors.append(f"{name}: Invalid configuration - {e}")
                    
        except yaml.YAMLError as e:
            errors.append(f"YAML parsing error: {e}")
        except Exception as e:
            errors.append(f"Configuration validation error: {e}")
            
        return len(errors) == 0, errors


# Example configuration file
EXAMPLE_CONFIG = """
# Model deployment configuration for Foundry
models:
  teacher:
    model_id: "microsoft/phi-2"
    deployment_type: "local_docker"
    port: 8001
    gpu_memory: 80  # percentage
    max_batch_size: 32
    max_sequence_length: 2048
    environment:
      CUDA_VISIBLE_DEVICES: "0"
    health_check_endpoint: "/v1/models"
    startup_timeout: 300

  student:
    model_id: "microsoft/phi-2"
    deployment_type: "local_docker"
    port: 8002
    gpu_memory: 80
    max_batch_size: 32
    max_sequence_length: 2048
    environment:
      CUDA_VISIBLE_DEVICES: "1"

  remote_teacher:
    model_id: "meta-llama/Llama-2-7b-hf"
    deployment_type: "remote_ssh"
    port: 8003
    environment:
      REMOTE_HOST: "gpu-server.example.com"
      REMOTE_PORT: "8000"
    health_check_endpoint: "/v1/models"
    startup_timeout: 60  # Assuming already running
"""


if __name__ == "__main__":
    # Example usage
    manager = ModelDeploymentManager()
    
    # Deploy a model
    deployment = manager.deploy_model("teacher")
    print(f"Deployed: {deployment.config.name} - Status: {deployment.status.value}")
    
    # Check status
    status = manager.get_deployment_status("teacher")
    print(f"Status: {json.dumps(status, indent=2)}")
    
    # Validate deployment
    valid, error = manager.validate_deployment("teacher")
    print(f"Validation: {'Success' if valid else f'Failed - {error}'}")