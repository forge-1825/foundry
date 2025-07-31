"""
Pytest configuration and shared fixtures for Foundry test suite
"""
import pytest
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Common test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory"""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    return TEST_DATA_DIR


@pytest.fixture
def sample_pdf_content():
    """Sample PDF text content for testing"""
    return """
    This is a sample PDF content for testing purposes.
    It contains multiple paragraphs and sentences.
    
    The content discusses various topics including:
    - Machine learning concepts
    - Natural language processing
    - Model distillation techniques
    
    This allows us to test extraction and enrichment functionality.
    """


@pytest.fixture
def sample_enriched_data():
    """Sample enriched data structure"""
    return [
        {
            "url": "test_document.pdf",
            "text": "This is sample text from a PDF document.",
            "content": "This is sample text from a PDF document.",
            "summary": "Sample text from PDF.",
            "entities": ["PDF"],
            "keywords": ["sample", "text", "document"],
            "metadata": {"pages": 1, "source": "test"}
        }
    ]


@pytest.fixture
def sample_teacher_pairs():
    """Sample teacher Q&A pairs"""
    return [
        {
            "input": "What is model distillation?",
            "target": "Model distillation is a technique where a smaller student model learns from a larger teacher model."
        },
        {
            "input": "Why use model distillation?",
            "target": "Model distillation creates smaller, faster models while preserving most of the teacher's performance."
        }
    ]


@pytest.fixture
def mock_vllm_response():
    """Mock response from vLLM API"""
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1234567890,
        "model": "test-model",
        "choices": [
            {
                "text": "This is a mock response from the model.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for tests"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir


# Skip markers for conditional test execution
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring model containers"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Auto-skip tests based on environment
def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on environment"""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_model = pytest.mark.skip(reason="Model containers not running")
    
    for item in items:
        # Skip GPU tests if CUDA not available
        if "requires_gpu" in item.keywords:
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)
        
        # Skip model tests if containers not running
        if "requires_model" in item.keywords:
            import socket
            try:
                # Check if model server is running on port 8000
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', 8000))
                sock.close()
                if result != 0:
                    item.add_marker(skip_model)
            except:
                item.add_marker(skip_model)