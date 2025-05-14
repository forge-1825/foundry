from fastapi import APIRouter, HTTPException
import psutil
import platform
import socket
import os
import logging
import subprocess
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create router
system_bp = APIRouter(prefix="/api/system", tags=["system"])

def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return []
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 6:
                    name, total, used, free, temp, util = parts[:6]
                    
                    # Convert to integers
                    try:
                        total = int(float(total)) * 1024 * 1024  # Convert MiB to bytes
                        used = int(float(used)) * 1024 * 1024
                        free = int(float(free)) * 1024 * 1024
                        temp = int(float(temp))
                        util = int(float(util))
                        
                        memory_percent = (used / total) * 100 if total > 0 else 0
                        
                        gpu_info.append({
                            "name": name,
                            "memory_total": total,
                            "memory_used": used,
                            "memory_free": free,
                            "memory_percent": memory_percent,
                            "temperature": temp,
                            "utilization": util
                        })
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error parsing GPU info: {e}")
        
        return gpu_info
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return []

def get_python_info():
    """Get Python environment information"""
    try:
        # Get Python version
        python_version = platform.python_version()
        
        # Check if running in a virtual environment
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            virtual_env = os.path.basename(virtual_env)
        
        # Get installed packages
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True
        )
        
        packages_count = 0
        if result.returncode == 0:
            try:
                packages = json.loads(result.stdout)
                packages_count = len(packages)
            except json.JSONDecodeError:
                packages_count = 0
        
        return {
            "version": python_version,
            "virtual_env": virtual_env,
            "packages": packages_count
        }
    except Exception as e:
        logger.error(f"Error getting Python info: {e}")
        return {
            "version": platform.python_version(),
            "virtual_env": None,
            "packages": 0
        }

@system_bp.get("/status")
async def get_system_status():
    """Get system status information"""
    try:
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical_count = psutil.cpu_count(logical=False)
        
        # Try to get CPU model
        cpu_model = "Unknown"
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        cpu_model = lines[1].strip()
            elif platform.system() == "Linux":
                result = subprocess.run(
                    ["cat", "/proc/cpuinfo"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if "model name" in line:
                            cpu_model = line.split(':')[1].strip()
                            break
            elif platform.system() == "Darwin":  # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    cpu_model = result.stdout.strip()
        except Exception as e:
            logger.error(f"Error getting CPU model: {e}")
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk information
        disk = psutil.disk_usage('/')
        
        # System information
        system_info = {
            "os": f"{platform.system()} {platform.release()}",
            "hostname": socket.gethostname(),
            "uptime": int(datetime.now().timestamp() - psutil.boot_time())
        }
        
        # GPU information
        gpu_info = get_gpu_info()
        
        # Python information
        python_info = get_python_info()
        
        return {
            "cpu_percent": cpu_percent,
            "cpu_info": {
                "model": cpu_model,
                "cores": cpu_count,
                "physical_cores": cpu_physical_count
            },
            "memory_percent": memory_percent,
            "memory_info": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "free": memory.free
            },
            "disk_info": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            },
            "system_info": system_info,
            "gpu_info": gpu_info,
            "python_info": python_info
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@system_bp.get("/processes")
async def get_system_processes():
    """Get running processes"""
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent', 'create_time']):
            try:
                proc_info = proc.info
                proc_info['create_time'] = datetime.fromtimestamp(proc_info['create_time']).isoformat()
                processes.append(proc_info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        # Sort by CPU usage (descending)
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        
        return processes[:50]  # Return top 50 processes
    except Exception as e:
        logger.error(f"Error getting system processes: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting system processes: {str(e)}")

@system_bp.get("/network")
async def get_network_info():
    """Get network information"""
    try:
        network_info = {}
        net_io = psutil.net_io_counters(pernic=True)
        
        for interface, stats in net_io.items():
            network_info[interface] = {
                "bytes_sent": stats.bytes_sent,
                "bytes_recv": stats.bytes_recv,
                "packets_sent": stats.packets_sent,
                "packets_recv": stats.packets_recv,
                "errin": stats.errin,
                "errout": stats.errout,
                "dropin": stats.dropin,
                "dropout": stats.dropout
            }
        
        return network_info
    except Exception as e:
        logger.error(f"Error getting network info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting network info: {str(e)}")
