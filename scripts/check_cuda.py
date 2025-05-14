#!/usr/bin/env python
"""
CUDA Availability Check

This script checks if CUDA is available through PyTorch and provides
information about the CUDA environment.

Usage:
    python check_cuda.py [--verbose]
"""

import argparse
import sys
import subprocess
import re
import torch

# Define the base safe_print function first
def safe_print(text):
    """
    Print text with Unicode characters safely.

    Args:
        text: The text to print
    """
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace Unicode characters with their ASCII equivalents
        ascii_text = text.encode('ascii', 'replace').decode('ascii')
        print(ascii_text)

# Define safe_print(name):
    if name.startswith('handle_unicode'):
        return safe_print
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Apply our custom __getattr__ to this module
sys.modules[__name__].__getattr__ = __getattr__






























    if verbose:
        safe_print(f"Running command: {cmd}")
    
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if verbose:
            safe_print(f"Command output:\n{result.stdout}")
            if result.stderr:
                safe_print(f"Command error:\n{result.stderr}")
        
        return result
    except subprocess.SubprocessError as e:
        if verbose:
            safe_print(f"Command failed: {e}")
        return None

def check_pytorch_installation(verbose=False):
    """Check if PyTorch is installed and if CUDA is available."""
    try:
        
        
        if verbose:
            safe_print(f"PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            
            safe_print(f"CUDA is available (version {cuda_version})")
            safe_print(f"Device count: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                safe_print(f"Device {i}: {device_name}")
            
            return True, cuda_version, device_count
        else:
            safe_print("CUDA is not available through PyTorch")
            return False, None, 0
    except ImportError:
        safe_print("PyTorch is not installed")
        return False, None, 0

def check_nvidia_smi(verbose=False):
    """Check if nvidia-smi is available and get GPU information."""
    result = run_command("nvidia-smi", verbose)
    
    if result and result.returncode == 0:
        safe_print("nvidia-smi is available")
        
        # Extract CUDA version from nvidia-smi output
        cuda_version_match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
        if cuda_version_match:
            cuda_version = cuda_version_match.group(1)
            safe_print(f"CUDA version from nvidia-smi: {cuda_version}")
        else:
            cuda_version = None
        
        # Extract GPU information from nvidia-smi output
        gpu_info = []
        gpu_lines = re.findall(r"\|\s+\d+\s+(\d+)\s+[^|]+\|[^|]+\|[^|]+\|", result.stdout)
        
        if gpu_lines:
            safe_print(f"Found {len(gpu_lines)} GPUs")
        
        return True, cuda_version, len(gpu_lines)
    else:
        safe_print("nvidia-smi is not available")
        return False, None, 0

def check_nvcc(verbose=False):
    """Check if nvcc (NVIDIA CUDA Compiler) is available."""
    result = run_command("nvcc --version", verbose)
    
    if result and result.returncode == 0:
        safe_print("nvcc is available")
        
        # Extract CUDA version from nvcc output
        cuda_version_match = re.search(r"release (\d+\.\d+)", result.stdout)
        if cuda_version_match:
            cuda_version = cuda_version_match.group(1)
            safe_print(f"CUDA version from nvcc: {cuda_version}")
            return True, cuda_version
        else:
            return True, None
    else:
        safe_print("nvcc is not available")
        return False, None

def main():
    parser = argparse.ArgumentParser(description="Check CUDA availability")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    
    safe_print("=== CUDA Availability Check ===")
    
    # Check PyTorch CUDA availability
    safe_print("\nChecking PyTorch CUDA availability...")
    pytorch_available, pytorch_cuda_version, pytorch_device_count = check_pytorch_installation(args.verbose)
    
    # Check nvidia-smi
    safe_print("\nChecking nvidia-smi...")
    nvidia_smi_available, nvidia_smi_cuda_version, nvidia_smi_device_count = check_nvidia_smi(args.verbose)
    
    # Check nvcc
    safe_print("\nChecking nvcc...")
    nvcc_available, nvcc_cuda_version = check_nvcc(args.verbose)
    
    # Summary
    safe_print("\n=== Summary ===")
    
    if pytorch_available and pytorch_cuda_version:
        safe_print(f"[WHITE HEAVY CHECK MARK] PyTorch CUDA: Available (version {pytorch_cuda_version}, {pytorch_device_count} devices)")
    else:
        safe_print("[CROSS MARK] PyTorch CUDA: Not available")
    
    if nvidia_smi_available:
        if nvidia_smi_cuda_version:
            safe_print(f"[WHITE HEAVY CHECK MARK] nvidia-smi: Available (CUDA version {nvidia_smi_cuda_version}, {nvidia_smi_device_count} GPUs)")
        else:
            safe_print(f"[WHITE HEAVY CHECK MARK] nvidia-smi: Available ({nvidia_smi_device_count} GPUs)")
    else:
        safe_print("[CROSS MARK] nvidia-smi: Not available")
    
    if nvcc_available:
        if nvcc_cuda_version:
            safe_print(f"[WHITE HEAVY CHECK MARK] nvcc: Available (CUDA version {nvcc_cuda_version})")
        else:
            safe_print("[WHITE HEAVY CHECK MARK] nvcc: Available")
    else:
        safe_print("[CROSS MARK] nvcc: Not available")
    
    # Overall CUDA availability
    cuda_available = pytorch_available or nvidia_smi_available or nvcc_available
    
    if cuda_available:
        safe_print("\n[WHITE HEAVY CHECK MARK] CUDA is available on this system")
        return 0
    else:
        safe_print("\n[CROSS MARK] CUDA is not available on this system")
        safe_print("vLLM requires CUDA for GPU acceleration")
        return 1

if __name__ == "__main__":
    sys.exit(main())
