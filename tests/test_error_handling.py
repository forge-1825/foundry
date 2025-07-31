"""
Test cases for unified error handling framework
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from backend.utils.unified_error_handler import (
    FoundryError, ConfigurationError, ModelError, NetworkError,
    ResourceError, ValidationError, ErrorSeverity, ErrorCategory,
    ErrorLogger, ErrorRecoveryManager, handle_errors, with_retry,
    error_context, format_error_response
)


class TestFoundryError:
    """Test FoundryError base class"""
    
    def test_error_creation(self):
        """Test creating a FoundryError"""
        error = FoundryError(
            message="Test error",
            error_code="TEST_001",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            details={"key": "value"},
            recoverable=True,
            retry_after=30
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_001"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.severity == ErrorSeverity.ERROR
        assert error.details == {"key": "value"}
        assert error.recoverable is True
        assert error.retry_after == 30
        assert error.timestamp is not None
        
    def test_error_to_dict(self):
        """Test converting error to dictionary"""
        error = FoundryError(
            message="Test error",
            error_code="TEST_001",
            category=ErrorCategory.MODEL
        )
        
        error_dict = error.to_dict()
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "TEST_001"
        assert error_dict["category"] == "model"
        assert "timestamp" in error_dict


class TestSpecificErrors:
    """Test specific error types"""
    
    def test_configuration_error(self):
        """Test ConfigurationError"""
        error = ConfigurationError("Config missing", {"file": "config.yaml"})
        assert error.error_code == "CFG_001"
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.details["file"] == "config.yaml"
        
    def test_model_error(self):
        """Test ModelError"""
        error = ModelError("Model failed", "gpt-4", {"reason": "timeout"})
        assert error.error_code == "MDL_001"
        assert error.category == ErrorCategory.MODEL
        assert error.details["model_name"] == "gpt-4"
        assert error.details["reason"] == "timeout"
        
    def test_network_error(self):
        """Test NetworkError"""
        error = NetworkError("Connection failed", "http://api.example.com")
        assert error.error_code == "NET_001"
        assert error.category == ErrorCategory.NETWORK
        assert error.recoverable is True
        assert error.retry_after == 5
        assert error.details["endpoint"] == "http://api.example.com"
        
    def test_resource_error(self):
        """Test ResourceError"""
        error = ResourceError("GPU out of memory", "gpu")
        assert error.error_code == "RES_001"
        assert error.category == ErrorCategory.RESOURCE
        assert error.recoverable is True
        assert error.retry_after == 30
        assert error.details["resource_type"] == "gpu"
        
    def test_validation_error(self):
        """Test ValidationError"""
        error = ValidationError("Invalid format", "email")
        assert error.error_code == "VAL_001"
        assert error.category == ErrorCategory.VALIDATION
        assert error.details["field"] == "email"


class TestErrorLogger:
    """Test ErrorLogger functionality"""
    
    @pytest.fixture
    def temp_log_dir(self, tmp_path):
        """Create temporary log directory"""
        return tmp_path / "logs"
        
    def test_error_logger_creation(self, temp_log_dir):
        """Test creating ErrorLogger"""
        logger = ErrorLogger("test", temp_log_dir)
        assert logger.log_dir == temp_log_dir
        assert temp_log_dir.exists()
        assert logger.error_counts == {}
        
    def test_log_error(self, temp_log_dir):
        """Test logging an error"""
        logger = ErrorLogger("test", temp_log_dir)
        
        error = FoundryError(
            message="Test error",
            error_code="TEST_001",
            category=ErrorCategory.MODEL
        )
        
        context = {"user": "test_user", "action": "test_action"}
        logger.log_error(error, context)
        
        # Check error counts
        assert logger.error_counts["model:TEST_001"] == 1
        
        # Check detailed log file exists
        log_files = list(temp_log_dir.glob("errors_detailed_*.jsonl"))
        assert len(log_files) == 1
        
        # Read and verify log entry
        with open(log_files[0], 'r') as f:
            log_entry = json.loads(f.readline())
            assert log_entry["error"]["message"] == "Test error"
            assert log_entry["context"]["user"] == "test_user"
            
    def test_error_statistics(self, temp_log_dir):
        """Test error statistics"""
        logger = ErrorLogger("test", temp_log_dir)
        
        # Log multiple errors
        for i in range(3):
            error = ModelError(f"Error {i}", "model1")
            logger.log_error(error)
            
        for i in range(2):
            error = NetworkError(f"Error {i}", "endpoint1")
            logger.log_error(error)
            
        stats = logger.get_error_statistics()
        assert stats["total_errors"] == 5
        assert "model" in stats["error_categories"]
        assert "network" in stats["error_categories"]


class TestErrorRecoveryManager:
    """Test ErrorRecoveryManager"""
    
    def test_recovery_strategy_registration(self):
        """Test registering recovery strategies"""
        manager = ErrorRecoveryManager()
        
        def test_strategy(error: FoundryError) -> bool:
            return error.details.get("recoverable", False)
            
        manager.register_recovery_strategy("TEST_001", test_strategy)
        
        # Test recoverable error
        error1 = FoundryError(
            message="Recoverable",
            error_code="TEST_001",
            recoverable=True,
            details={"recoverable": True}
        )
        assert manager.can_recover(error1) is True
        
        # Test non-recoverable error
        error2 = FoundryError(
            message="Not recoverable",
            error_code="TEST_001",
            recoverable=True,
            details={"recoverable": False}
        )
        assert manager.can_recover(error2) is False
        
    def test_retry_policy(self):
        """Test retry policy"""
        manager = ErrorRecoveryManager()
        manager.register_retry_policy("NET_001", max_retries=3, backoff_factor=2.0)
        
        error = NetworkError("Connection failed", "endpoint")
        
        # First retry
        should_retry, wait_time = manager.should_retry(error)
        assert should_retry is True
        assert wait_time == 5  # Uses error's retry_after
        
        # Second retry
        should_retry, wait_time = manager.should_retry(error)
        assert should_retry is True
        
        # Third retry
        should_retry, wait_time = manager.should_retry(error)
        assert should_retry is True
        
        # Fourth retry (exceeds max)
        should_retry, wait_time = manager.should_retry(error)
        assert should_retry is False


class TestDecorators:
    """Test error handling decorators"""
    
    def test_handle_errors_decorator(self):
        """Test @handle_errors decorator"""
        @handle_errors(error_class=ValueError, error_code="TEST_001")
        def test_function(value):
            if value < 0:
                raise ValueError("Negative value")
            return value * 2
            
        # Normal execution
        assert test_function(5) == 10
        
        # Error handling
        with pytest.raises(FoundryError) as exc_info:
            test_function(-1)
            
        error = exc_info.value
        assert error.error_code == "TEST_001"
        assert "Negative value" in error.message
        
    def test_with_retry_decorator(self):
        """Test @with_retry decorator"""
        attempt_count = 0
        
        @with_retry(max_retries=2, backoff_factor=0.1)
        def flaky_function():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise NetworkError("Connection failed", "test")
            return "success"
            
        # Should succeed on third attempt
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3
        
    def test_with_retry_failure(self):
        """Test @with_retry when all retries fail"""
        @with_retry(max_retries=2, backoff_factor=0.1)
        def always_fails():
            raise NetworkError("Connection failed", "test")
            
        with pytest.raises(NetworkError):
            always_fails()


class TestErrorContext:
    """Test error context manager"""
    
    def test_error_context_success(self):
        """Test error_context with successful execution"""
        with error_context("test_operation", {"key": "value"}) as context:
            assert context["context_name"] == "test_operation"
            assert context["context_data"]["key"] == "value"
            assert "start_time" in context
            
    def test_error_context_with_error(self):
        """Test error_context with error"""
        with pytest.raises(FoundryError):
            with error_context("test_operation", {"key": "value"}):
                raise ValueError("Test error")


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_format_error_response(self):
        """Test format_error_response"""
        error = NetworkError("Connection failed", "endpoint")
        response = format_error_response(error)
        
        assert response["error"] is True
        assert response["error_code"] == "NET_001"
        assert response["message"] == "Connection failed"
        assert response["category"] == "network"
        assert response["recoverable"] is True
        assert response["retry_after"] == 5
        
    @patch.dict('os.environ', {'FOUNDRY_ENV': 'development'})
    def test_format_error_response_development(self):
        """Test format_error_response in development mode"""
        error = ModelError("Model failed", "gpt-4", {"reason": "timeout"})
        response = format_error_response(error)
        
        assert "details" in response
        assert response["details"]["model_name"] == "gpt-4"


@pytest.mark.integration
class TestIntegration:
    """Integration tests for error handling"""
    
    def test_full_error_flow(self, temp_log_dir):
        """Test complete error handling flow"""
        from backend.utils.unified_error_handler import (
            initialize_error_handling, get_error_logger, get_recovery_manager
        )
        
        # Initialize error handling
        initialize_error_handling("test", temp_log_dir)
        
        # Create and log an error
        error = NetworkError("API timeout", "http://api.example.com/data")
        logger = get_error_logger()
        logger.log_error(error, {"request_id": "12345"})
        
        # Check recovery
        manager = get_recovery_manager()
        assert manager.can_recover(error) is True
        
        # Check retry
        should_retry, wait_time = manager.should_retry(error)
        assert should_retry is True
        assert wait_time > 0
        
        # Check statistics
        stats = logger.get_error_statistics()
        assert stats["total_errors"] == 1
        assert "network" in stats["error_categories"]