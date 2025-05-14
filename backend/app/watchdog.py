import asyncio
import logging
import time
import gc
import os
import signal
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable

# Try to import GPU monitoring libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ProcessWatchdog:
    """
    A watchdog class that monitors processes for inactivity, high memory usage, and low GPU utilization.
    Can automatically take recovery actions when issues are detected.
    """
    
    def __init__(
        self,
        timeout_seconds: int = 300,  # 5 minutes
        memory_threshold: float = 90.0,  # 90% GPU memory usage
        utilization_threshold: float = 10.0,  # 10% GPU utilization
        check_interval_seconds: int = 60,  # Check every minute
        auto_recovery: bool = True
    ):
        self.timeout_seconds = timeout_seconds
        self.memory_threshold = memory_threshold
        self.utilization_threshold = utilization_threshold
        self.check_interval_seconds = check_interval_seconds
        self.auto_recovery = auto_recovery
        
        # Process tracking
        self.last_activity: Dict[str, datetime] = {}
        self.script_info: Dict[str, Dict[str, Any]] = {}
        self.stuck_processes: Dict[str, Dict[str, Any]] = {}
        self.recovery_attempts: Dict[str, int] = {}
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Recovery callbacks
        self.recovery_callbacks: Dict[str, Callable] = {}
    
    def reset_script_tracking(self, script_id: str) -> None:
        """Reset tracking for a script."""
        self.last_activity[script_id] = datetime.now()
        self.script_info[script_id] = {
            "memory_usage": None,
            "gpu_utilization": None,
            "progress": None
        }
        if script_id in self.stuck_processes:
            del self.stuck_processes[script_id]
        self.recovery_attempts[script_id] = 0
    
    def update_activity(self, script_id: str, progress_info: Optional[Dict[str, Any]] = None) -> None:
        """Update the last activity time for a script."""
        self.last_activity[script_id] = datetime.now()
        
        # Update script info if progress info is provided
        if progress_info:
            if script_id not in self.script_info:
                self.script_info[script_id] = {
                    "memory_usage": None,
                    "gpu_utilization": None,
                    "progress": None
                }
            
            # Update memory usage if available
            if "memory_usage" in progress_info:
                self.script_info[script_id]["memory_usage"] = progress_info["memory_usage"]
            
            # Update GPU utilization if available
            if "gpu_utilization" in progress_info:
                self.script_info[script_id]["gpu_utilization"] = progress_info["gpu_utilization"]
            
            # Update progress if available
            if "percent" in progress_info:
                self.script_info[script_id]["progress"] = progress_info["percent"]
        
        # If the script was previously stuck, remove it from stuck processes
        if script_id in self.stuck_processes:
            logger.info(f"Script {script_id} is no longer stuck")
            del self.stuck_processes[script_id]
    
    def register_recovery_callback(self, action: str, callback: Callable) -> None:
        """Register a callback function for a recovery action."""
        self.recovery_callbacks[action] = callback
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Started process watchdog monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread is not running")
            return
        
        self.stop_monitoring.set()
        self.monitoring_thread.join(timeout=5.0)
        if self.monitoring_thread.is_alive():
            logger.warning("Monitoring thread did not stop cleanly")
        else:
            logger.info("Stopped process watchdog monitoring")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        while not self.stop_monitoring.is_set():
            try:
                self._check_processes()
                
                # Sleep for the specified interval
                self.stop_monitoring.wait(self.check_interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                # Sleep a bit to avoid tight loop in case of persistent errors
                time.sleep(1.0)
    
    def _check_processes(self) -> None:
        """Check all monitored processes for issues."""
        now = datetime.now()
        
        # Get current GPU memory usage and utilization
        gpu_memory_percent = None
        gpu_utilization = None
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_memory_percent = gpu.memoryUtil * 100
                    gpu_utilization = gpu.load * 100
                    logger.debug(f"GPU Memory: {gpu_memory_percent:.1f}%, Utilization: {gpu_utilization:.1f}%")
            except Exception as e:
                logger.error(f"Error getting GPU info: {e}")
        
        # Check each script
        for script_id, last_activity_time in list(self.last_activity.items()):
            # Skip scripts that are already marked as stuck
            if script_id in self.stuck_processes:
                continue
            
            # Check for timeout
            time_since_activity = now - last_activity_time
            if time_since_activity.total_seconds() > self.timeout_seconds:
                logger.warning(f"Script {script_id} has been inactive for {time_since_activity.total_seconds():.1f} seconds")
                self._mark_script_as_stuck(script_id, "inactivity", {
                    "timeout_seconds": self.timeout_seconds,
                    "time_since_activity": time_since_activity.total_seconds()
                })
                continue
            
            # Check for high memory usage with low utilization
            if gpu_memory_percent is not None and gpu_utilization is not None:
                if gpu_memory_percent > self.memory_threshold and gpu_utilization < self.utilization_threshold:
                    logger.warning(f"Script {script_id} may have a memory leak: GPU Memory {gpu_memory_percent:.1f}%, Utilization {gpu_utilization:.1f}%")
                    self._mark_script_as_stuck(script_id, "memory_leak", {
                        "gpu_memory_percent": gpu_memory_percent,
                        "gpu_utilization": gpu_utilization,
                        "memory_threshold": self.memory_threshold,
                        "utilization_threshold": self.utilization_threshold
                    })
                    continue
    
    def _mark_script_as_stuck(self, script_id: str, reason: str, details: Dict[str, Any]) -> None:
        """Mark a script as stuck and take recovery action if auto-recovery is enabled."""
        # Initialize recovery attempts counter if not already present
        if script_id not in self.recovery_attempts:
            self.recovery_attempts[script_id] = 0
        
        # Mark as stuck
        self.stuck_processes[script_id] = {
            "reason": reason,
            "details": details,
            "detected_at": datetime.now().isoformat(),
            "recovery_attempts": self.recovery_attempts[script_id]
        }
        
        # Take recovery action if auto-recovery is enabled
        if self.auto_recovery:
            self._take_recovery_action(script_id, reason)
    
    def _take_recovery_action(self, script_id: str, reason: str) -> None:
        """Take appropriate recovery action based on the issue."""
        # Increment recovery attempts counter
        self.recovery_attempts[script_id] += 1
        attempts = self.recovery_attempts[script_id]
        
        logger.info(f"Taking recovery action for script {script_id} (attempt {attempts})")
        
        # Different actions based on number of attempts
        if attempts == 1:
            # First attempt: Force garbage collection
            self.force_garbage_collection(script_id)
        elif attempts == 2:
            # Second attempt: Clear CUDA cache
            self.clear_cuda_cache(script_id)
        elif attempts == 3:
            # Third attempt: Restart the script
            self.restart_script(script_id)
        else:
            # More than three attempts: Log but don't take further action
            logger.error(f"Script {script_id} is still stuck after {attempts} recovery attempts")
    
    def force_garbage_collection(self, script_id: str) -> None:
        """Force Python garbage collection."""
        logger.info(f"Forcing garbage collection for script {script_id}")
        gc.collect()
        
        # Call registered callback if available
        if "force_gc" in self.recovery_callbacks:
            try:
                self.recovery_callbacks["force_gc"](script_id)
            except Exception as e:
                logger.error(f"Error in force_gc callback: {e}")
    
    def clear_cuda_cache(self, script_id: str) -> None:
        """Clear CUDA cache if PyTorch is available."""
        logger.info(f"Clearing CUDA cache for script {script_id}")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}")
        else:
            logger.warning("PyTorch or CUDA not available, cannot clear CUDA cache")
        
        # Call registered callback if available
        if "clear_cuda" in self.recovery_callbacks:
            try:
                self.recovery_callbacks["clear_cuda"](script_id)
            except Exception as e:
                logger.error(f"Error in clear_cuda callback: {e}")
    
    def restart_script(self, script_id: str) -> None:
        """Restart the script."""
        logger.info(f"Attempting to restart script {script_id}")
        
        # Call registered callback if available
        if "restart" in self.recovery_callbacks:
            try:
                self.recovery_callbacks["restart"](script_id)
                logger.info(f"Script {script_id} restart initiated")
            except Exception as e:
                logger.error(f"Error restarting script: {e}")
        else:
            logger.warning(f"No restart callback registered, cannot restart script {script_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the watchdog."""
        return {
            "monitored_processes": list(self.last_activity.keys()),
            "stuck_processes": self.stuck_processes,
            "settings": {
                "timeout_seconds": self.timeout_seconds,
                "memory_threshold": self.memory_threshold,
                "utilization_threshold": self.utilization_threshold,
                "check_interval_seconds": self.check_interval_seconds,
                "auto_recovery": self.auto_recovery
            }
        }
    
    def perform_action(self, action: str, script_id: str) -> Dict[str, Any]:
        """Perform a manual recovery action."""
        logger.info(f"Manual action requested: {action} for script {script_id}")
        
        if action == "force_gc":
            self.force_garbage_collection(script_id)
            return {"success": True, "message": "Garbage collection forced"}
        elif action == "clear_cuda":
            self.clear_cuda_cache(script_id)
            return {"success": True, "message": "CUDA cache cleared"}
        elif action == "restart":
            self.restart_script(script_id)
            return {"success": True, "message": "Script restart initiated"}
        elif action == "reset":
            self.reset_script_tracking(script_id)
            return {"success": True, "message": "Script tracking reset"}
        else:
            logger.error(f"Unknown action: {action}")
            return {"success": False, "message": f"Unknown action: {action}"}

# Async function to run the watchdog service
async def run_watchdog_service(watchdog: ProcessWatchdog) -> None:
    """Run the watchdog service in the background."""
    watchdog.start_monitoring()
    
    try:
        # Keep the service running
        while True:
            await asyncio.sleep(60)  # Check every minute
    except asyncio.CancelledError:
        # Stop the watchdog when the task is cancelled
        watchdog.stop_monitoring()
        raise
    except Exception as e:
        logger.error(f"Error in watchdog service: {e}")
        watchdog.stop_monitoring()
        raise

# Example usage
if __name__ == "__main__":
    # Create a watchdog instance
    watchdog = ProcessWatchdog(
        timeout_seconds=60,  # 1 minute for testing
        memory_threshold=80.0,
        utilization_threshold=20.0,
        check_interval_seconds=10  # Check every 10 seconds for testing
    )
    
    # Register a restart callback
    def restart_callback(script_id: str) -> None:
        print(f"Restarting script {script_id}...")
    
    watchdog.register_recovery_callback("restart", restart_callback)
    
    # Start monitoring
    watchdog.start_monitoring()
    
    # Simulate a script
    script_id = "test_script"
    watchdog.reset_script_tracking(script_id)
    
    try:
        # Simulate activity
        for i in range(10):
            print(f"Script activity {i+1}/10")
            watchdog.update_activity(script_id, {"percent": (i+1) * 10})
            time.sleep(5)
        
        # Simulate inactivity
        print("Simulating inactivity...")
        time.sleep(70)  # Longer than timeout_seconds
        
        # Should have been marked as stuck by now
        print("Watchdog status:", watchdog.get_status())
    finally:
        # Stop monitoring
        watchdog.stop_monitoring()
