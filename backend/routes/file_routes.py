from fastapi import APIRouter, HTTPException
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create router
file_bp = APIRouter(prefix="/api/files", tags=["files"])

@file_bp.get("/browse")
async def browse_directory(path: Optional[str] = None):
    """Browse directory contents"""
    try:
        # If no path is provided, use the current directory
        if not path:
            path = os.getcwd()
        
        # Normalize path
        path = os.path.normpath(path)
        
        # Check if path exists
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Path not found: {path}")
        
        # Check if path is a directory
        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")
        
        # Get directory contents
        contents = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            is_dir = os.path.isdir(item_path)
            
            try:
                # Get file/directory stats
                stats = os.stat(item_path)
                
                contents.append({
                    "name": item,
                    "path": item_path,
                    "is_directory": is_dir,
                    "size": stats.st_size if not is_dir else None,
                    "modified": stats.st_mtime
                })
            except Exception as e:
                logger.error(f"Error getting stats for {item_path}: {e}")
                # Include the item with limited information
                contents.append({
                    "name": item,
                    "path": item_path,
                    "is_directory": is_dir,
                    "size": None,
                    "modified": None,
                    "error": str(e)
                })
        
        # Sort contents: directories first, then files, both alphabetically
        contents.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
        
        # Get parent directory
        parent_dir = os.path.dirname(path) if path != os.path.dirname(path) else None
        
        return {
            "current_path": path,
            "parent_directory": parent_dir,
            "contents": contents
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error browsing directory: {e}")
        raise HTTPException(status_code=500, detail=f"Error browsing directory: {str(e)}")

@file_bp.get("/drives")
async def get_drives():
    """Get available drives (Windows only)"""
    try:
        if os.name != 'nt':
            return {"drives": []}
        
        import win32api
        
        drives = []
        for drive in win32api.GetLogicalDriveStrings().split('\000')[:-1]:
            try:
                drive_type = win32api.GetDriveType(drive)
                drive_types = {
                    0: "Unknown",
                    1: "No Root Directory",
                    2: "Removable",
                    3: "Fixed",
                    4: "Network",
                    5: "CD-ROM",
                    6: "RAM Disk"
                }
                
                # Get volume information
                try:
                    volume_name, volume_serial, max_component_length, file_system_flags, file_system_name = win32api.GetVolumeInformation(drive)
                except:
                    volume_name = ""
                    file_system_name = ""
                
                drives.append({
                    "path": drive,
                    "type": drive_types.get(drive_type, "Unknown"),
                    "name": volume_name,
                    "file_system": file_system_name
                })
            except Exception as e:
                logger.error(f"Error getting drive info for {drive}: {e}")
                drives.append({
                    "path": drive,
                    "type": "Unknown",
                    "error": str(e)
                })
        
        return {"drives": drives}
    except Exception as e:
        logger.error(f"Error getting drives: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting drives: {str(e)}")

@file_bp.get("/exists")
async def check_path_exists(path: str):
    """Check if a path exists"""
    try:
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False
        is_file = os.path.isfile(path) if exists else False
        
        return {
            "exists": exists,
            "is_directory": is_dir,
            "is_file": is_file
        }
    except Exception as e:
        logger.error(f"Error checking path: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking path: {str(e)}")

@file_bp.get("/default-paths")
async def get_default_paths():
    """Get default paths for the application"""
    try:
        # Get current working directory
        cwd = os.getcwd()
        
        # Get user home directory
        home = str(Path.home())
        
        # Get documents directory
        documents = os.path.join(home, "Documents")
        
        # Get desktop directory
        desktop = os.path.join(home, "Desktop")
        
        # Get downloads directory
        downloads = os.path.join(home, "Downloads")
        
        # Get application data directory
        app_data = os.path.join(cwd, "data")
        
        return {
            "current_working_directory": cwd,
            "home_directory": home,
            "documents_directory": documents,
            "desktop_directory": desktop,
            "downloads_directory": downloads,
            "application_data_directory": app_data
        }
    except Exception as e:
        logger.error(f"Error getting default paths: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting default paths: {str(e)}")
