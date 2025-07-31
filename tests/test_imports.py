"""
Smoke tests to verify all pipeline components can import without errors
"""
import pytest
import importlib
import sys
from pathlib import Path


@pytest.mark.smoke
class TestImports:
    """Test that all scripts can be imported successfully"""
    
    def test_import_data_enrichment(self):
        """Test importing data enrichment script"""
        try:
            import data_enrichment
            assert hasattr(data_enrichment, 'main')
        except ImportError as e:
            pytest.fail(f"Failed to import data_enrichment: {e}")
    
    def test_import_teacher_pair_generation(self):
        """Test importing teacher pair generation script"""
        try:
            import teacher_pair_generation_vllm_hierarchical
            assert hasattr(teacher_pair_generation_vllm_hierarchical, 'main')
        except ImportError as e:
            pytest.fail(f"Failed to import teacher_pair_generation_vllm_hierarchical: {e}")
    
    def test_import_distillation(self):
        """Test importing distillation script"""
        try:
            import distillation_vllm_faster_improved
            # This script may not have a main function
        except ImportError as e:
            pytest.fail(f"Failed to import distillation_vllm_faster_improved: {e}")
    
    def test_import_merge_model(self):
        """Test importing merge model script"""
        try:
            import merge_model
            assert hasattr(merge_model, 'merge_lora_to_base_model')
        except ImportError as e:
            pytest.fail(f"Failed to import merge_model: {e}")
    
    def test_import_student_self_study(self):
        """Test importing student self study script"""
        try:
            import student_self_study_enhanced
            assert hasattr(student_self_study_enhanced, 'main')
        except ImportError as e:
            pytest.fail(f"Failed to import student_self_study_enhanced: {e}")
    
    def test_import_evaluation(self):
        """Test importing evaluation script"""
        try:
            import run_openevals_langchain
            # Check for expected functions/classes
        except ImportError as e:
            pytest.fail(f"Failed to import run_openevals_langchain: {e}")
    
    def test_import_manual_extractor(self):
        """Test importing manual extractor script"""
        try:
            import manual_extractor
            assert hasattr(manual_extractor, 'main')
        except ImportError as e:
            pytest.fail(f"Failed to import manual_extractor: {e}")
    
    def test_import_memory_monitor(self):
        """Test importing memory monitor script"""
        try:
            import memory_monitor
            assert hasattr(memory_monitor, 'MemoryMonitor')
        except ImportError as e:
            pytest.fail(f"Failed to import memory_monitor: {e}")
    
    def test_import_backend_modules(self):
        """Test importing backend modules"""
        backend_modules = [
            'app.main',
            'app.vllm_client',
            'config.pipeline_config',
            'routes.pipeline_routes',
            'utils.error_handler',
            'utils.process_manager'
        ]
        
        for module in backend_modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"Failed to import {module}: {e}")


@pytest.mark.smoke
class TestDependencies:
    """Test that required dependencies are available"""
    
    def test_pytorch_available(self):
        """Test PyTorch is available"""
        try:
            import torch
            assert torch.__version__
        except ImportError:
            pytest.skip("PyTorch not installed")
    
    def test_transformers_available(self):
        """Test Transformers is available"""
        try:
            import transformers
            assert transformers.__version__
        except ImportError:
            pytest.fail("Transformers library not installed")
    
    def test_fastapi_available(self):
        """Test FastAPI is available"""
        try:
            import fastapi
            assert fastapi.__version__
        except ImportError:
            pytest.fail("FastAPI not installed")
    
    def test_pydantic_available(self):
        """Test Pydantic is available"""
        try:
            import pydantic
            assert pydantic.__version__
        except ImportError:
            pytest.fail("Pydantic not installed")
    
    def test_numpy_available(self):
        """Test NumPy is available"""
        try:
            import numpy
            assert numpy.__version__
        except ImportError:
            pytest.fail("NumPy not installed")
    
    def test_pandas_available(self):
        """Test Pandas is available"""
        try:
            import pandas
            assert pandas.__version__
        except ImportError:
            pytest.skip("Pandas not installed")
    
    def test_requests_available(self):
        """Test Requests is available"""
        try:
            import requests
            assert requests.__version__
        except ImportError:
            pytest.fail("Requests not installed")