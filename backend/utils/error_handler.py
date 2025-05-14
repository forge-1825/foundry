from functools import wraps
from flask import jsonify
import os

class PipelineError(Exception):
    def __init__(self, message, error_code=500, details=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

def handle_pipeline_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except PipelineError as e:
            return jsonify({
                'error': e.message,
                'details': e.details
            }), e.error_code
        except Exception as e:
            return jsonify({
                'error': 'An unexpected error occurred',
                'details': str(e)
            }), 500
    return decorated_function

def check_gpu_requirements():
    """Check if GPU requirements are met for the pipeline."""
    try:
        import torch
        if not torch.cuda.is_available():
            raise PipelineError(
                "GPU not available",
                error_code=400,
                details={'cuda_available': False}
            )
        return True
    except ImportError:
        raise PipelineError(
            "PyTorch not installed",
            error_code=500,
            details={'pytorch_installed': False}
        )

def validate_model_paths():
    """Validate that all required model paths exist."""
    required_paths = [
        'models/phi-4',
        'models/phi-2'
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        raise PipelineError(
            "Missing required model paths",
            error_code=400,
            details={'missing_paths': missing_paths}
        )
