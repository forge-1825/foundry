import os
import time
import threading
import json
import logging
import functools
import gc
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import GPU monitoring libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU memory monitoring will be limited")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logger.warning("GPUtil not available, GPU monitoring will be limited")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, CPU memory monitoring will be limited")

class MemorySnapshot:
    """A snapshot of memory usage at a specific point in time."""
    
    def __init__(self, script_id: str, step: Optional[str] = None):
        self.script_id = script_id
        self.timestamp = datetime.now().isoformat()
        self.step = step
        self.cpu_memory = self._get_cpu_memory() if PSUTIL_AVAILABLE else None
        self.gpu_memory = self._get_gpu_memory() if TORCH_AVAILABLE or GPUTIL_AVAILABLE else None
        self.gpu_utilization = self._get_gpu_utilization() if GPUTIL_AVAILABLE else None
    
    def _get_cpu_memory(self) -> Dict[str, float]:
        """Get CPU memory usage."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
            "percent": process.memory_percent(),  # Process memory as percentage of total
            "system_total_mb": system_memory.total / (1024 * 1024),  # Total system memory in MB
            "system_used_mb": system_memory.used / (1024 * 1024),  # Used system memory in MB
            "system_percent": system_memory.percent  # System memory usage percentage
        }
    
    def _get_gpu_memory(self) -> Optional[Dict[str, Any]]:
        """Get GPU memory usage."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # Get memory usage for each GPU
                gpu_memory = []
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)  # MB
                    reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)  # MB
                    max_memory = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)  # MB
                    
                    gpu_memory.append({
                        "device": i,
                        "name": torch.cuda.get_device_name(i),
                        "allocated_mb": allocated,
                        "reserved_mb": reserved,
                        "total_mb": max_memory,
                        "percent": (allocated / max_memory) * 100 if max_memory > 0 else 0
                    })
                
                return {
                    "devices": gpu_memory,
                    "total_allocated_mb": sum(gpu["allocated_mb"] for gpu in gpu_memory),
                    "total_reserved_mb": sum(gpu["reserved_mb"] for gpu in gpu_memory)
                }
            except Exception as e:
                logger.error(f"Error getting PyTorch GPU memory: {e}")
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_memory = []
                    for gpu in gpus:
                        gpu_memory.append({
                            "device": gpu.id,
                            "name": gpu.name,
                            "used_mb": gpu.memoryUsed,
                            "total_mb": gpu.memoryTotal,
                            "percent": gpu.memoryUtil * 100,
                            "temperature": gpu.temperature
                        })
                    
                    return {
                        "devices": gpu_memory,
                        "total_used_mb": sum(gpu["used_mb"] for gpu in gpu_memory)
                    }
            except Exception as e:
                logger.error(f"Error getting GPUtil GPU memory: {e}")
        
        return None
    
    def _get_gpu_utilization(self) -> Optional[List[Dict[str, Any]]]:
        """Get GPU utilization (compute load)."""
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return [
                        {
                            "device": gpu.id,
                            "name": gpu.name,
                            "utilization_percent": gpu.load * 100,
                            "temperature": gpu.temperature
                        }
                        for gpu in gpus
                    ]
            except Exception as e:
                logger.error(f"Error getting GPU utilization: {e}")
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the snapshot to a dictionary."""
        return {
            "script_id": self.script_id,
            "timestamp": self.timestamp,
            "step": self.step,
            "cpu_memory": self.cpu_memory,
            "gpu_memory": self.gpu_memory,
            "gpu_utilization": self.gpu_utilization
        }
    
    def log(self) -> None:
        """Log the memory snapshot."""
        gpu_info = ""
        if self.gpu_memory and "devices" in self.gpu_memory and self.gpu_memory["devices"]:
            first_gpu = self.gpu_memory["devices"][0]
            gpu_info = f"GPU Memory: {first_gpu.get('used_mb', 0):.1f}MB ({first_gpu.get('percent', 0):.1f}%)"
            
            if self.gpu_utilization and len(self.gpu_utilization) > 0:
                gpu_info += f", Utilization: {self.gpu_utilization[0].get('utilization_percent', 0):.1f}%"
        
        cpu_info = ""
        if self.cpu_memory:
            cpu_info = f"CPU Memory: {self.cpu_memory.get('rss_mb', 0):.1f}MB ({self.cpu_memory.get('percent', 0):.1f}%)"
        
        step_info = f" [{self.step}]" if self.step else ""
        
        logger.info(f"Memory snapshot{step_info}: {gpu_info} {cpu_info}")

def log_memory_snapshot(script_id: str, step: Optional[str] = None) -> MemorySnapshot:
    """Take and log a memory snapshot."""
    snapshot = MemorySnapshot(script_id, step)
    snapshot.log()
    return snapshot

class MemoryMonitor:
    """
    A class to monitor memory usage over time.
    Can be used as a decorator, context manager, or standalone monitor.
    """
    
    def __init__(self, script_id: str, interval_seconds: float = 5.0, log_to_file: bool = True):
        self.script_id = script_id
        self.interval_seconds = interval_seconds
        self.log_to_file = log_to_file
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.log_file = f"{script_id}_memory.json" if log_to_file else None
    
    def start(self) -> None:
        """Start monitoring memory usage in a background thread."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Memory monitoring is already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info(f"Started memory monitoring for {self.script_id} (interval: {self.interval_seconds}s)")
    
    def stop(self) -> None:
        """Stop monitoring memory usage."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Memory monitoring is not running")
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=self.interval_seconds + 1.0)
        if self.monitoring_thread.is_alive():
            logger.warning("Memory monitoring thread did not stop cleanly")
        else:
            logger.info(f"Stopped memory monitoring for {self.script_id}")
        
        # Save snapshots to file
        if self.log_to_file and self.log_file and self.snapshots:
            self._save_snapshots()
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in a background thread."""
        while not self.stop_monitoring.is_set():
            try:
                # Take a memory snapshot
                snapshot = MemorySnapshot(self.script_id)
                self.snapshots.append(snapshot)
                snapshot.log()
                
                # Sleep for the specified interval
                self.stop_monitoring.wait(self.interval_seconds)
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                # Sleep a bit to avoid tight loop in case of persistent errors
                time.sleep(1.0)
    
    def _save_snapshots(self) -> None:
        """Save memory snapshots to a JSON file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump([snapshot.to_dict() for snapshot in self.snapshots], f, indent=2)
            logger.info(f"Saved {len(self.snapshots)} memory snapshots to {self.log_file}")
        except Exception as e:
            logger.error(f"Error saving memory snapshots to file: {e}")
    
    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator to monitor memory during function execution."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
    
    def __enter__(self) -> 'MemoryMonitor':
        """Use as a context manager to monitor memory during a block of code."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop monitoring when exiting the context."""
        self.stop()

@contextmanager
def track_memory(script_id: str, step: Optional[str] = None):
    """
    A context manager to track memory usage before and after a block of code.
    
    Example:
    ```
    with track_memory("my_script", "loading_data"):
        data = load_large_dataset()
    ```
    """
    # Force garbage collection before measuring
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Take before snapshot
    before = log_memory_snapshot(script_id, f"{step} (before)" if step else "before")
    
    try:
        yield
    finally:
        # Force garbage collection after the block
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Take after snapshot
        after = log_memory_snapshot(script_id, f"{step} (after)" if step else "after")
        
        # Calculate and log the difference
        if before.gpu_memory and after.gpu_memory and "total_allocated_mb" in before.gpu_memory and "total_allocated_mb" in after.gpu_memory:
            diff_mb = after.gpu_memory["total_allocated_mb"] - before.gpu_memory["total_allocated_mb"]
            logger.info(f"Memory change during {step if step else 'operation'}: {diff_mb:.1f}MB")

def memory_usage_decorator(script_id: str, step: Optional[str] = None):
    """
    A decorator to track memory usage before and after a function call.
    
    Example:
    ```
    @memory_usage_decorator("my_script", "process_batch")
    def process_batch(batch):
        # Process the batch
        return result
    ```
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with track_memory(script_id, step or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Example 1: Take a one-time snapshot
    snapshot = log_memory_snapshot("example", "initial_check")
    
    # Example 2: Use as a context manager for a specific block
    with track_memory("example", "loading_data"):
        # Simulate loading data
        data = [i for i in range(1000000)]
        time.sleep(1)
    
    # Example 3: Use as a decorator
    @memory_usage_decorator("example", "processing")
    def process_data(data):
        # Process the data
        result = [i * 2 for i in data]
        time.sleep(1)
        return result
    
    processed = process_data(data)
    
    # Example 4: Continuous monitoring
    monitor = MemoryMonitor("example", interval_seconds=1.0)
    monitor.start()
    
    try:
        # Simulate some memory-intensive operations
        for i in range(5):
            # Allocate some memory
            more_data = [i for i in range((i + 1) * 1000000)]
            time.sleep(1)
            
            # Free some memory
            more_data = None
            gc.collect()
            time.sleep(1)
    finally:
        # Stop monitoring
        monitor.stop()
