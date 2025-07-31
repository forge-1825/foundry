"""
Foundry Test Framework
Comprehensive testing infrastructure for the Foundry AI system
"""

import os
import sys
import json
import pytest
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import requests
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FoundryTestBase:
    """Base class for all Foundry tests"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix="foundry_test_")
        cls.data_dir = Path(cls.test_dir) / "data"
        cls.output_dir = Path(cls.test_dir) / "output"
        cls.models_dir = Path(cls.test_dir) / "models"
        
        # Create directories
        for dir_path in [cls.data_dir, cls.output_dir, cls.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Set environment variables
        os.environ['FOUNDRY_TEST_MODE'] = '1'
        os.environ['FOUNDRY_DATA_DIR'] = str(cls.data_dir)
        os.environ['FOUNDRY_OUTPUT_DIR'] = str(cls.output_dir)
        
    @classmethod
    def teardown_class(cls):
        """Cleanup test environment"""
        if hasattr(cls, 'test_dir') and os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            
        # Clean environment
        os.environ.pop('FOUNDRY_TEST_MODE', None)
        os.environ.pop('FOUNDRY_DATA_DIR', None)
        os.environ.pop('FOUNDRY_OUTPUT_DIR', None)
    
    def create_test_data(self, filename: str, data: Dict[str, Any]) -> Path:
        """Create test data file"""
        file_path = self.data_dir / filename
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return file_path
    
    def read_output_file(self, filename: str) -> Dict[str, Any]:
        """Read output file"""
        file_path = self.output_dir / filename
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def mock_vllm_response(self, text: str) -> Dict[str, Any]:
        """Create mock vLLM response"""
        return {
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }


class ModelMockFactory:
    """Factory for creating model mocks"""
    
    @staticmethod
    def create_vllm_client_mock():
        """Create mock vLLM client"""
        mock_client = Mock()
        mock_client.query_model.return_value = {
            "choices": [{
                "text": "Mock response",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }]
        }
        mock_client.get_available_models.return_value = [
            {"id": "mock-model", "object": "model"}
        ]
        return mock_client
    
    @staticmethod
    def create_torch_model_mock():
        """Create mock PyTorch model"""
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_model.config = Mock(max_length=512)
        return mock_model
    
    @staticmethod
    def create_tokenizer_mock():
        """Create mock tokenizer"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Decoded text"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        return mock_tokenizer


class PipelineTestFixtures:
    """Test fixtures for pipeline components"""
    
    @staticmethod
    def sample_extraction_data():
        """Sample data for manual extraction"""
        return {
            "documents": [
                {
                    "id": "doc1",
                    "title": "Test Document",
                    "content": "This is a test document for extraction.",
                    "metadata": {"source": "test"}
                }
            ]
        }
    
    @staticmethod
    def sample_enrichment_data():
        """Sample data for enrichment"""
        return {
            "extracted_data": [
                {
                    "id": "item1",
                    "text": "Original text",
                    "concepts": ["concept1", "concept2"]
                }
            ]
        }
    
    @staticmethod
    def sample_teacher_pair_data():
        """Sample data for teacher pair generation"""
        return {
            "enriched_data": [
                {
                    "id": "item1",
                    "question": "What is the purpose?",
                    "context": "The purpose is testing.",
                    "answer": "Testing the system"
                }
            ]
        }
    
    @staticmethod
    def sample_distillation_data():
        """Sample data for distillation"""
        return {
            "teacher_pairs": [
                {
                    "input": "Question about the system",
                    "teacher_output": "Detailed teacher response",
                    "student_output": "Student attempt"
                }
            ]
        }


class IntegrationTestBase(FoundryTestBase):
    """Base class for integration tests"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.api_base_url = os.environ.get('FOUNDRY_API_URL', 'http://localhost:5000')
        self.session = requests.Session()
        
    def teardown_method(self):
        """Cleanup after each test method"""
        self.session.close()
    
    def wait_for_service(self, url: str, timeout: int = 30) -> bool:
        """Wait for service to be available"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(url, timeout=1)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
            
        return False
    
    def start_pipeline(self, config: Dict[str, Any]) -> str:
        """Start a pipeline run"""
        response = self.session.post(
            f"{self.api_base_url}/api/pipeline/start",
            json=config
        )
        response.raise_for_status()
        return response.json()['pipeline_id']
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline status"""
        response = self.session.get(
            f"{self.api_base_url}/api/pipeline/status/{pipeline_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def wait_for_pipeline_completion(self, pipeline_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for pipeline to complete"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_pipeline_status(pipeline_id)
            if status['status'] in ['completed', 'failed']:
                return status
            time.sleep(5)
            
        raise TimeoutError(f"Pipeline {pipeline_id} did not complete within {timeout} seconds")


# Test configuration management
class TestConfig:
    """Test configuration management"""
    
    DEFAULT_CONFIG = {
        "test_mode": True,
        "mock_models": True,
        "gpu_required": False,
        "timeout": 60,
        "batch_size": 4,
        "log_level": "INFO"
    }
    
    @classmethod
    def load_config(cls, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load test configuration"""
        config = cls.DEFAULT_CONFIG.copy()
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                config.update(custom_config)
                
        # Override with environment variables
        for key, value in config.items():
            env_key = f"FOUNDRY_TEST_{key.upper()}"
            if env_key in os.environ:
                config[key] = os.environ[env_key]
                
        return config


# Pytest fixtures
@pytest.fixture
def test_base():
    """Provide test base instance"""
    base = FoundryTestBase()
    base.setup_class()
    yield base
    base.teardown_class()


@pytest.fixture
def mock_vllm_client():
    """Provide mock vLLM client"""
    return ModelMockFactory.create_vllm_client_mock()


@pytest.fixture
def mock_torch_model():
    """Provide mock PyTorch model"""
    return ModelMockFactory.create_torch_model_mock()


@pytest.fixture
def test_fixtures():
    """Provide test fixtures"""
    return PipelineTestFixtures()


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return TestConfig.load_config()


# Custom pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests"
    )
    config.addinivalue_line(
        "markers", "model_required: Tests requiring actual models"
    )