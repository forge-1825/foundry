"""
Integration tests for model connectivity
"""
import pytest
import requests
import socket
import time
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


@pytest.mark.integration
@pytest.mark.requires_model
class TestModelConnectivity:
    """Test connectivity to model servers"""
    
    def test_teacher_model_connectivity(self):
        """Test connection to teacher model server"""
        try:
            response = requests.get("http://localhost:8000/v1/models", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert len(data["data"]) > 0
        except requests.exceptions.ConnectionError:
            pytest.skip("Teacher model server not running on port 8000")
        except Exception as e:
            pytest.fail(f"Unexpected error connecting to teacher model: {e}")
    
    def test_student_model_connectivity(self):
        """Test connection to student model server"""
        try:
            response = requests.get("http://localhost:8002/v1/models", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("Student model server not running on port 8002")
        except Exception as e:
            pytest.fail(f"Unexpected error connecting to student model: {e}")
    
    def test_distilled_model_connectivity(self):
        """Test connection to distilled model server"""
        try:
            response = requests.get("http://localhost:8003/v1/models", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("Distilled model server not running on port 8003")
        except Exception as e:
            pytest.fail(f"Unexpected error connecting to distilled model: {e}")
    
    def test_model_inference_teacher(self):
        """Test inference on teacher model"""
        try:
            payload = {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
                "temperature": 0.7
            }
            response = requests.post(
                "http://localhost:8000/v1/completions",
                json=payload,
                timeout=30
            )
            assert response.status_code == 200
            data = response.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            assert "text" in data["choices"][0]
        except requests.exceptions.ConnectionError:
            pytest.skip("Teacher model server not available")
        except Exception as e:
            pytest.fail(f"Unexpected error during teacher inference: {e}")
    
    def test_vllm_client_detection(self):
        """Test vLLM client server detection"""
        try:
            from app.vllm_client import detect_all_vllm_servers
            servers = detect_all_vllm_servers()
            assert isinstance(servers, list)
            # Check structure of detected servers
            for server in servers:
                assert "name" in server
                assert "port" in server
                assert "status" in server
                assert "type" in server
        except ImportError:
            pytest.fail("Failed to import vLLM client")
        except Exception as e:
            pytest.fail(f"Error detecting vLLM servers: {e}")
    
    def test_remote_model_support(self):
        """Test remote model environment variable"""
        use_remote = os.environ.get("USE_REMOTE_MODELS", "false").lower() == "true"
        if use_remote:
            assert True  # Remote models configured
        else:
            # Check if SSH tunnels would be needed
            ports_to_check = [8000, 8002, 8003]
            local_models = 0
            for port in ports_to_check:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    if result == 0:
                        local_models += 1
                except:
                    pass
            
            if local_models == 0:
                pytest.skip("No local models found and USE_REMOTE_MODELS not set")


@pytest.mark.integration
class TestBackendConnectivity:
    """Test backend API connectivity"""
    
    def test_backend_health_check(self):
        """Test backend health endpoint"""
        try:
            response = requests.get("http://localhost:7433/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data.get("status") == "healthy"
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running on port 7433")
        except Exception as e:
            pytest.fail(f"Unexpected error connecting to backend: {e}")
    
    def test_backend_models_endpoint(self):
        """Test backend models detection endpoint"""
        try:
            response = requests.get("http://localhost:7433/api/models", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"Error accessing models endpoint: {e}")
    
    def test_backend_pipeline_steps(self):
        """Test backend pipeline steps endpoint"""
        try:
            response = requests.get("http://localhost:7433/api/pipeline/steps", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) > 0
            # Check structure
            for step in data:
                assert "id" in step
                assert "name" in step
                assert "description" in step
                assert "script" in step
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")
        except Exception as e:
            pytest.fail(f"Error accessing pipeline steps: {e}")