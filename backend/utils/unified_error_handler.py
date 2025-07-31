"""
Unified Error Handling Framework for Foundry
Provides standardized error handling, logging, and recovery mechanisms
"""

import os
import sys
import json
import logging
import traceback
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from functools import wraps
from contextlib import contextmanager
import threading
import queue

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    CONFIGURATION = "configuration"
    NETWORK = "network"
    MODEL = "model"
    DATA = "data"
    RESOURCE = "resource"
    PERMISSION = "permission"
    VALIDATION = "validation"
    PIPELINE = "pipeline"
    UNKNOWN = "unknown"


class FoundryError(Exception):
    """Base exception class for all Foundry errors"""
    
    def __init__(self, 
                 message: str,
                 error_code: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 details: Optional[Dict[str, Any]] = None,
                 recoverable: bool = False,
                 retry_after: Optional[int] = None):
        """
        Initialize Foundry error
        
        Args:
            message: Human-readable error message
            error_code: Unique error code (e.g., "E001", "PIPE_001")
            category: Error category for classification
            severity: Error severity level
            details: Additional error details
            recoverable: Whether the error is recoverable
            retry_after: Seconds to wait before retry (if recoverable)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.utcnow().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp
        }


# Specific error types
class ConfigurationError(FoundryError):
    """Configuration-related errors"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CFG_001",
            category=ErrorCategory.CONFIGURATION,
            details=details
        )


class ModelError(FoundryError):
    """Model-related errors"""
    def __init__(self, message: str, model_name: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["model_name"] = model_name
        super().__init__(
            message=message,
            error_code="MDL_001",
            category=ErrorCategory.MODEL,
            details=details
        )


class ResourceError(FoundryError):
    """Resource-related errors (GPU, memory, etc.)"""
    def __init__(self, message: str, resource_type: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["resource_type"] = resource_type
        super().__init__(
            message=message,
            error_code="RES_001",
            category=ErrorCategory.RESOURCE,
            details=details,
            recoverable=True,
            retry_after=30
        )


class NetworkError(FoundryError):
    """Network-related errors"""
    def __init__(self, message: str, endpoint: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["endpoint"] = endpoint
        super().__init__(
            message=message,
            error_code="NET_001",
            category=ErrorCategory.NETWORK,
            details=details,
            recoverable=True,
            retry_after=5
        )


class ValidationError(FoundryError):
    """Data validation errors"""
    def __init__(self, message: str, field: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["field"] = field
        super().__init__(
            message=message,
            error_code="VAL_001",
            category=ErrorCategory.VALIDATION,
            details=details
        )


class ErrorLogger:
    """Centralized error logging with context"""
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file handler for errors
        error_log_path = self.log_dir / f"{name}_errors.log"
        file_handler = logging.FileHandler(error_log_path)
        file_handler.setLevel(logging.ERROR)
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra_context)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Error statistics
        self.error_counts = {}
        self.error_queue = queue.Queue()
        
    def log_error(self, error: FoundryError, context: Optional[Dict[str, Any]] = None):
        """Log error with context"""
        # Update error counts
        error_key = f"{error.category.value}:{error.error_code}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Create log entry
        log_entry = {
            "error": error.to_dict(),
            "context": context or {},
            "stack_trace": traceback.format_exc(),
            "thread_id": threading.current_thread().ident,
            "process_id": os.getpid()
        }
        
        # Log based on severity
        extra = {"extra_context": json.dumps(context or {})}
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error.message, extra=extra)
        elif error.severity == ErrorSeverity.ERROR:
            self.logger.error(error.message, extra=extra)
        elif error.severity == ErrorSeverity.WARNING:
            self.logger.warning(error.message, extra=extra)
        else:
            self.logger.info(error.message, extra=extra)
            
        # Queue for async processing
        self.error_queue.put(log_entry)
        
        # Write detailed error log
        self._write_detailed_log(log_entry)
        
    def _write_detailed_log(self, log_entry: Dict[str, Any]):
        """Write detailed error log to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        detailed_log_path = self.log_dir / f"errors_detailed_{timestamp}.jsonl"
        
        with open(detailed_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "error_counts": self.error_counts,
            "total_errors": sum(self.error_counts.values()),
            "error_categories": list(set(k.split(':')[0] for k in self.error_counts.keys()))
        }


class ErrorRecoveryManager:
    """Manages error recovery strategies"""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.retry_policies = {}
        
    def register_recovery_strategy(self, 
                                 error_code: str,
                                 strategy: Callable[[FoundryError], bool]):
        """Register a recovery strategy for an error code"""
        self.recovery_strategies[error_code] = strategy
        
    def register_retry_policy(self,
                            error_code: str,
                            max_retries: int = 3,
                            backoff_factor: float = 2.0):
        """Register retry policy for an error code"""
        self.retry_policies[error_code] = {
            "max_retries": max_retries,
            "backoff_factor": backoff_factor,
            "current_retry": 0
        }
        
    def can_recover(self, error: FoundryError) -> bool:
        """Check if error can be recovered"""
        if not error.recoverable:
            return False
            
        if error.error_code in self.recovery_strategies:
            return self.recovery_strategies[error.error_code](error)
            
        return False
        
    def should_retry(self, error: FoundryError) -> Tuple[bool, Optional[int]]:
        """Check if operation should be retried"""
        if error.error_code not in self.retry_policies:
            return False, None
            
        policy = self.retry_policies[error.error_code]
        if policy["current_retry"] >= policy["max_retries"]:
            return False, None
            
        policy["current_retry"] += 1
        wait_time = error.retry_after or (policy["backoff_factor"] ** policy["current_retry"])
        
        return True, int(wait_time)


# Global error management
_error_logger = None
_recovery_manager = None


def initialize_error_handling(name: str = "foundry", log_dir: Optional[Path] = None):
    """Initialize global error handling"""
    global _error_logger, _recovery_manager
    _error_logger = ErrorLogger(name, log_dir)
    _recovery_manager = ErrorRecoveryManager()
    
    # Register default recovery strategies
    def network_recovery(error: FoundryError) -> bool:
        # Simple network recovery - just return True to allow retry
        return True
        
    def resource_recovery(error: FoundryError) -> bool:
        # Check if resources might be available
        if "gpu" in error.details.get("resource_type", "").lower():
            # Could check GPU availability here
            return True
        return True
        
    _recovery_manager.register_recovery_strategy("NET_001", network_recovery)
    _recovery_manager.register_recovery_strategy("RES_001", resource_recovery)
    
    # Register default retry policies
    _recovery_manager.register_retry_policy("NET_001", max_retries=3, backoff_factor=2.0)
    _recovery_manager.register_retry_policy("RES_001", max_retries=5, backoff_factor=3.0)


def get_error_logger() -> ErrorLogger:
    """Get global error logger"""
    global _error_logger
    if _error_logger is None:
        initialize_error_handling()
    return _error_logger


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get global recovery manager"""
    global _recovery_manager
    if _recovery_manager is None:
        initialize_error_handling()
    return _recovery_manager


# Decorators for error handling
def handle_errors(error_class: type = FoundryError,
                 error_code: str = "GEN_001",
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 log_context: bool = True):
    """Decorator for automatic error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = {
                "function": func.__name__,
                "module": func.__module__,
                "args": str(args)[:200],  # Limit size
                "kwargs": str(kwargs)[:200]
            }
            
            try:
                return func(*args, **kwargs)
            except error_class as e:
                if log_context:
                    get_error_logger().log_error(e, context)
                raise
            except Exception as e:
                # Convert to FoundryError
                foundry_error = FoundryError(
                    message=str(e),
                    error_code=error_code,
                    category=category,
                    details={"original_error": type(e).__name__}
                )
                if log_context:
                    get_error_logger().log_error(foundry_error, context)
                raise foundry_error
                
        return wrapper
    return decorator


def with_retry(max_retries: int = 3,
              backoff_factor: float = 2.0,
              retriable_errors: List[type] = None):
    """Decorator for automatic retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retriable = retriable_errors or [NetworkError, ResourceError]
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(retriable) as e:
                    last_error = e
                    if attempt < max_retries:
                        wait_time = e.retry_after or (backoff_factor ** attempt)
                        logger = get_error_logger()
                        logger.logger.info(
                            f"Retrying {func.__name__} after {wait_time}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        import time
                        time.sleep(wait_time)
                    else:
                        raise
                        
            if last_error:
                raise last_error
                
        return wrapper
    return decorator


@contextmanager
def error_context(context_name: str, context_data: Dict[str, Any] = None):
    """Context manager for error handling with context"""
    context = {
        "context_name": context_name,
        "context_data": context_data or {},
        "start_time": datetime.utcnow().isoformat()
    }
    
    try:
        yield context
    except FoundryError as e:
        context["end_time"] = datetime.utcnow().isoformat()
        get_error_logger().log_error(e, context)
        raise
    except Exception as e:
        context["end_time"] = datetime.utcnow().isoformat()
        foundry_error = FoundryError(
            message=str(e),
            error_code="CTX_001",
            category=ErrorCategory.UNKNOWN,
            details={"original_error": type(e).__name__}
        )
        get_error_logger().log_error(foundry_error, context)
        raise foundry_error


# Utility functions
def format_error_response(error: FoundryError) -> Dict[str, Any]:
    """Format error for API response"""
    response = {
        "error": True,
        "error_code": error.error_code,
        "message": error.message,
        "category": error.category.value,
        "timestamp": error.timestamp
    }
    
    if error.recoverable:
        response["recoverable"] = True
        if error.retry_after:
            response["retry_after"] = error.retry_after
            
    # Include details in development mode
    if os.environ.get("FOUNDRY_ENV") == "development":
        response["details"] = error.details
        
    return response


def log_performance_metric(operation: str, duration: float, success: bool = True):
    """Log performance metrics"""
    logger = get_error_logger()
    metric = {
        "operation": operation,
        "duration_seconds": duration,
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Log to performance file
    perf_log_path = logger.log_dir / "performance_metrics.jsonl"
    with open(perf_log_path, 'a') as f:
        f.write(json.dumps(metric) + '\n')