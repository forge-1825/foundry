from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import os
import asyncio
import logging
import json
import subprocess
import psutil
import GPUtil
import re
from datetime import datetime
from pathlib import Path

# Import the watchdog module
from app.watchdog import ProcessWatchdog, run_watchdog_service

# Import routes
# from routes.evaluation_routes import evaluation_bp  # Flask blueprint - needs conversion
from routes.pipeline_routes import pipeline_bp
from routes.docker_routes import docker_bp
# from routes.system_routes import system_bp  # Flask blueprint - needs conversion
from routes.file_routes import file_bp

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Model Distillation Pipeline API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register blueprints
# app.include_router(evaluation_bp)  # Commented - Flask blueprint
app.include_router(pipeline_bp)
app.include_router(docker_bp)
# app.include_router(system_bp)  # Commented - Flask blueprint
app.include_router(file_bp)

# Add a root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Model Distillation Pipeline API is running"}

# Constants
SCRIPTS_DIR = os.environ.get("SCRIPTS_DIR", "/scripts")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
ENABLE_GPU = os.environ.get("ENABLE_GPU", "1") == "1"

# Model Configuration from Environment
MODEL_CONFIG = {
    'teacher': {
        'name': os.environ.get("TEACHER_MODEL_NAME", "Teacher"),
        'port': int(os.environ.get("TEACHER_MODEL_PORT", "8000")),
        'host': os.environ.get("TEACHER_MODEL_HOST", "localhost"),
        'id': os.environ.get("TEACHER_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct-AWQ"),
        'path': os.environ.get("TEACHER_MODEL_PATH", "/models/teacher"),
        'remote': os.environ.get("TEACHER_MODEL_REMOTE", "True").lower() == "true"
    },
    'student': {
        'name': os.environ.get("STUDENT_MODEL_NAME", "Student"),
        'port': int(os.environ.get("STUDENT_MODEL_PORT", "8001")),
        'host': os.environ.get("STUDENT_MODEL_HOST", "localhost"),
        'id': os.environ.get("STUDENT_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct-AWQ"),
        'path': os.environ.get("STUDENT_MODEL_PATH", "/models/student"),
        'remote': os.environ.get("STUDENT_MODEL_REMOTE", "True").lower() == "true"
    },
    'distilled': {
        'name': os.environ.get("DISTILLED_MODEL_NAME", "Distilled"),
        'port': int(os.environ.get("DISTILLED_MODEL_PORT", "8003")),
        'host': os.environ.get("DISTILLED_MODEL_HOST", "localhost"),
        'id': os.environ.get("DISTILLED_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct-AWQ"),
        'path': os.environ.get("DISTILLED_MODEL_PATH", "/models/distilled"),
        'remote': os.environ.get("DISTILLED_MODEL_REMOTE", "True").lower() == "true"
    }
}

# Active processes dictionary
active_processes: Dict[str, subprocess.Popen] = {}
process_logs: Dict[str, List[str]] = {}
connected_clients: Dict[str, List[WebSocket]] = {}
script_progress: Dict[str, Dict[str, Any]] = {}  # Track detailed progress for each script

# Initialize watchdog
watchdog = ProcessWatchdog(
    timeout_seconds=300,  # 5 minutes
    memory_threshold=90.0,  # 90% GPU memory usage
    utilization_threshold=10.0,  # 10% GPU utilization
    check_interval_seconds=60  # Check every minute
)

# Models
class ScriptConfig(BaseModel):
    script_id: str
    parameters: Dict[str, Any]

class ScriptStatus(BaseModel):
    script_id: str
    status: str  # "pending", "running", "completed", "error"
    progress_percent: float = 0.0
    current_step: Optional[str] = None
    steps_completed: int = 0
    total_steps: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    estimated_time_remaining: Optional[int] = None  # In seconds
    memory_usage: Optional[Dict[str, float]] = None

class SystemStatus(BaseModel):
    cpu_percent: float
    memory_percent: float
    gpu_info: Optional[List[Dict[str, Any]]] = None
    active_scripts: List[str]

class WatchdogAction(BaseModel):
    action: str  # "force_gc", "clear_cuda", "restart", "reset"
    script_id: Optional[str] = None

# Helper functions
def get_script_path(script_id: str) -> str:
    """Get the full path for a script by ID."""
    script_mapping = {
        "manual_extractor": "manual_extractor.py",
        "data_enrichment": "data_enrichment_enhanced_gpu_fixed_v2.py",
        "content_extraction_enrichment": "data_enrichment_enhanced_gpu_fixed_v2.py",
        "teacher_pair_generation": "teacher_pair_generation_vllm_ssh.py",
        "distillation": "distillation_vllm_faster_improved.py",
        "student_self_study": "student_self_study_enhanced.py",
        "merge_model": "merge_model.py",
        "evaluation": "evaluate_distilled.py"
    }

    if script_id not in script_mapping:
        raise ValueError(f"Unknown script ID: {script_id}")

    script_path = os.path.join(SCRIPTS_DIR, script_mapping[script_id])
    if not os.path.exists(script_path):
        raise ValueError(f"Script not found: {script_path}")

    return script_path

def validate_path(path: str) -> bool:
    """Validate if a path exists."""
    if not path:
        return False

    # Handle Windows paths when running in Docker
    if path.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
        # For Windows paths, we can't directly check if they exist in the Docker container
        # Instead, we'll assume they exist and let the script handle any errors
        logging.warning(f"Windows path detected: {path}. Cannot validate in Docker container.")
        return True

    # For paths that should be accessible within the container
    return os.path.exists(path)

def parse_progress_from_log(script_id: str, log_line: str) -> Optional[Dict[str, Any]]:
    """
    Parse progress information from log lines.
    Returns a dictionary with progress information if found, None otherwise.
    """
    progress_info = None

    # Different regex patterns for different scripts
    if script_id == "manual_extractor":
        # Example: "Processing 5/20 files (25%)"
        match = re.search(r"Processing (\d+)/(\d+) files \((\d+)%\)", log_line)
        if match:
            current, total, percent = match.groups()
            progress_info = {
                "current": int(current),
                "total": int(total),
                "percent": float(percent),
                "step": "Processing files"
            }
    elif script_id == "data_enrichment":
        # Example: "Enriching record 10/50 (20%)"
        match = re.search(r"Enriching record (\d+)/(\d+) \((\d+)%\)", log_line)
        if match:
            current, total, percent = match.groups()
            progress_info = {
                "current": int(current),
                "total": int(total),
                "percent": float(percent),
                "step": "Enriching records"
            }
    elif script_id == "teacher_pair_generation":
        # Example: "Processing records 15/100 (15%)"
        match = re.search(r"Processing records? (\d+)/(\d+) \((\d+)%\)", log_line)
        if match:
            current, total, percent = match.groups()
            progress_info = {
                "current": int(current),
                "total": int(total),
                "percent": float(percent),
                "step": "Generating teacher pairs"
            }
    elif script_id == "distillation":
        # Example: "Epoch 2/5, Step 50/100, Loss: 0.123"
        epoch_match = re.search(r"Epoch (\d+)/(\d+)", log_line)
        step_match = re.search(r"Step (\d+)/(\d+)", log_line)
        loss_match = re.search(r"Loss: ([\d\.]+)", log_line)

        if epoch_match and step_match:
            current_epoch, total_epochs = epoch_match.groups()
            current_step, total_steps = step_match.groups()

            # Calculate overall progress
            epoch_progress = (int(current_epoch) - 1) / int(total_epochs)
            step_progress = int(current_step) / int(total_steps)
            overall_progress = (epoch_progress + step_progress / int(total_epochs)) * 100

            progress_info = {
                "current_epoch": int(current_epoch),
                "total_epochs": int(total_epochs),
                "current_step": int(current_step),
                "total_steps": int(total_steps),
                "percent": overall_progress,
                "step": f"Epoch {current_epoch}/{total_epochs}, Step {current_step}/{total_steps}"
            }

            if loss_match:
                progress_info["loss"] = float(loss_match.group(1))
    elif script_id == "student_self_study":
        # Example: "Processing PDF 2/5 (40%)"
        pdf_match = re.search(r"Processing PDF (\d+)/(\d+) \((\d+)%\)", log_line)
        # Example: "Generating questions 15/30 (50%)"
        question_match = re.search(r"Generating questions (\d+)/(\d+) \((\d+)%\)", log_line)

        if pdf_match:
            current, total, percent = pdf_match.groups()
            progress_info = {
                "current": int(current),
                "total": int(total),
                "percent": float(percent),
                "step": f"Processing PDF {current}/{total}"
            }
        elif question_match:
            current, total, percent = question_match.groups()
            progress_info = {
                "current": int(current),
                "total": int(total),
                "percent": float(percent),
                "step": f"Generating questions {current}/{total}"
            }
    elif script_id == "merge_model":
        # Example: "Merging model: 30% complete"
        merge_match = re.search(r"Merging model: (\d+)% complete", log_line)
        # Example: "Loading adapter weights: 45% complete"
        adapter_match = re.search(r"Loading adapter weights: (\d+)% complete", log_line)
        # Example: "Saving merged model: 80% complete"
        save_match = re.search(r"Saving merged model: (\d+)% complete", log_line)

        if merge_match:
            percent = merge_match.group(1)
            progress_info = {
                "percent": float(percent),
                "step": "Merging model"
            }
        elif adapter_match:
            percent = adapter_match.group(1)
            progress_info = {
                "percent": float(percent),
                "step": "Loading adapter weights"
            }
        elif save_match:
            percent = save_match.group(1)
            progress_info = {
                "percent": float(percent),
                "step": "Saving merged model"
            }
    elif script_id == "evaluation":
        # Example: "Evaluating prompt 3/10 (30%)"
        match = re.search(r"Evaluating prompt (\d+)/(\d+) \((\d+)%\)", log_line)
        if match:
            current, total, percent = match.groups()
            progress_info = {
                "current": int(current),
                "total": int(total),
                "percent": float(percent),
                "step": "Evaluating prompts"
            }

    # Check for memory usage information in any script
    memory_match = re.search(r"GPU Memory: ([\d\.]+)MB", log_line)
    if memory_match and progress_info:
        progress_info["memory_usage"] = float(memory_match.group(1))

    return progress_info

def update_script_progress(script_id: str, progress_info: Dict[str, Any]) -> None:
    """Update the progress tracking for a script."""
    if script_id not in script_progress:
        script_progress[script_id] = {
            "progress_percent": 0.0,
            "current_step": None,
            "steps_completed": 0,
            "total_steps": 0,
            "start_time": datetime.now().isoformat(),
            "memory_usage": None
        }

    # Update progress information
    if "percent" in progress_info:
        script_progress[script_id]["progress_percent"] = progress_info["percent"]

    if "step" in progress_info:
        script_progress[script_id]["current_step"] = progress_info["step"]

    if "current" in progress_info and "total" in progress_info:
        script_progress[script_id]["steps_completed"] = progress_info["current"]
        script_progress[script_id]["total_steps"] = progress_info["total"]

    if "memory_usage" in progress_info:
        script_progress[script_id]["memory_usage"] = {
            "gpu_mb": progress_info["memory_usage"]
        }

    # Update watchdog to indicate activity
    watchdog.update_activity(script_id, progress_info)

async def run_script(script_id: str, parameters: Dict[str, Any]) -> None:
    """Run a script with the given parameters."""
    script_path = get_script_path(script_id)

    # Initialize progress tracking
    script_progress[script_id] = {
        "progress_percent": 0.0,
        "current_step": "Starting...",
        "steps_completed": 0,
        "total_steps": 0,
        "start_time": datetime.now().isoformat(),
        "memory_usage": None
    }

    # Prepare command with parameters
    if script_id == "content_extraction_enrichment":
        # Run the Python script directly
        cmd = ["python3", script_path]

        # Add parameters for the batch file
        # URL parameter
        if "url" in parameters and parameters["url"]:
            cmd.append(parameters["url"])
        else:
            cmd.append("")

        # Source folder parameter
        if "source_folder" in parameters and parameters["source_folder"]:
            cmd.append(parameters["source_folder"])
        else:
            cmd.append("")

        # Docker folder parameter
        if "docker_folder" in parameters and parameters["docker_folder"]:
            cmd.append(parameters["docker_folder"])
        else:
            cmd.append("")

        # Output directory parameter
        if "output_dir" in parameters and parameters["output_dir"]:
            cmd.append(parameters["output_dir"])
        else:
            cmd.append("Output")

        # Extract links parameter
        if "extract_links" in parameters:
            cmd.append(str(parameters["extract_links"]).lower())
        else:
            cmd.append("false")

        # Enable enrichment parameter
        if "enable_enrichment" in parameters:
            cmd.append(str(parameters["enable_enrichment"]).lower())
        else:
            cmd.append("false")

        # Input file parameter
        if "input_file" in parameters and parameters["input_file"]:
            cmd.append(parameters["input_file"])
        else:
            cmd.append("")

        # Output file parameter
        if "output_file" in parameters and parameters["output_file"]:
            cmd.append(parameters["output_file"])
        else:
            cmd.append("")

        # Entity extraction parameter
        if "enable_entity_extraction" in parameters:
            cmd.append(str(parameters["enable_entity_extraction"]).lower())
        else:
            cmd.append("false")

        # Summarization parameter
        if "enable_summarization" in parameters:
            cmd.append(str(parameters["enable_summarization"]).lower())
        else:
            cmd.append("false")

        # Keyword extraction parameter
        if "enable_keyword_extraction" in parameters:
            cmd.append(str(parameters["enable_keyword_extraction"]).lower())
        else:
            cmd.append("false")

        # GPU parameter
        if "use_gpu" in parameters:
            cmd.append(str(parameters["use_gpu"]).lower())
        else:
            cmd.append("false")

        logger.info(f"Executing content_extraction_enrichment with command: {' '.join(cmd)}")
    else:
        cmd = ["python3", script_path]

    # Special handling for content_extraction_enrichment
    if script_id == "content_extraction_enrichment" and False:  # Skip this block since we're using the batch file
        # This is a special case that combines manual_extractor and data_enrichment
        # First, execute manual_extractor
        extraction_cmd = ["python", os.path.join(SCRIPTS_DIR, "manual_extractor.py")]

        # Handle URL parameter
        if "url" in parameters and parameters["url"]:
            extraction_cmd.append(f"--url={parameters['url']}")

        # Handle source folder parameter
        has_source_folder = "source_folder" in parameters and parameters["source_folder"]
        has_docker_folder = "docker_folder" in parameters and parameters["docker_folder"]

        if has_source_folder:
            source_folder = parameters['source_folder']
            # If it's a Windows path, map it to a Docker volume path
            if source_folder.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                # For now, we'll just use the path as is and let the script handle any errors
                logger.warning(f"Using Windows path in Docker container: {source_folder}")
                # Escape backslashes in Windows paths
                source_folder = source_folder.replace('\\', '\\\\')
            extraction_cmd.append(f"--source-folder={source_folder}")
        elif has_docker_folder:
            docker_folder = parameters['docker_folder']
            logger.info(f"Using Docker folder path: {docker_folder}")
            extraction_cmd.append(f"--docker-folder={docker_folder}")

        # Handle output directory
        if "output_dir" in parameters and parameters["output_dir"]:
            extraction_cmd.append(f"--output-dir={parameters['output_dir']}")
        else:
            # Default output directory
            extraction_cmd.append(f"--output-dir={DATA_DIR}/extracted")

        # Handle extract links option
        if "extract_links" in parameters and parameters["extract_links"]:
            extraction_cmd.append("--extract-links")

        # Execute manual extraction
        logger.info(f"Executing extraction step: {' '.join(extraction_cmd)}")
        process_logs[script_id] = []
        process_logs[script_id].append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting extraction step...")

        extraction_process = subprocess.Popen(
            extraction_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Read and log output
        for line in iter(extraction_process.stdout.readline, ''):
            if not line:
                break
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [EXTRACTION] {line.strip()}"
            process_logs[script_id].append(log_entry)

            # Broadcast log to connected clients
            if script_id in connected_clients:
                for websocket in connected_clients[script_id]:
                    try:
                        await websocket.send_text(log_entry)
                    except Exception as e:
                        logger.error(f"Error sending to websocket: {e}")

        # Wait for extraction to complete
        extraction_return_code = extraction_process.wait()
        if extraction_return_code != 0:
            error_message = f"Extraction step failed with code {extraction_return_code}"
            logger.error(error_message)
            process_logs[script_id].append(f"[ERROR] {error_message}")
            raise ValueError(error_message)

        # If enrichment is enabled, execute data_enrichment
        if "enable_enrichment" in parameters and parameters["enable_enrichment"]:
            # Prepare enrichment command
            enrichment_cmd = ["python", os.path.join(SCRIPTS_DIR, "data_enrichment_enhanced_gpu_fixed_v2.py")]

            # Handle input file parameter
            if "input_file" in parameters and parameters["input_file"]:
                input_file = parameters["input_file"]
                if input_file.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                    logger.warning(f"Using Windows path in Docker container: {input_file}")
                    # Escape backslashes in Windows paths
                    input_file = input_file.replace('\\', '\\\\')
                enrichment_cmd.append(f"--input-file={input_file}")
            else:
                # Default input file from extraction output
                output_dir = parameters.get("output_dir", f"{DATA_DIR}/extracted")
                enrichment_cmd.append(f"--input-file={output_dir}/extracted_data.json")

            # Handle output file parameter
            if "output_file" in parameters and parameters["output_file"]:
                output_file = parameters["output_file"]
                if output_file.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                    logger.warning(f"Using Windows path in Docker container: {output_file}")
                    # Escape backslashes in Windows paths
                    output_file = output_file.replace('\\', '\\\\')
                enrichment_cmd.append(f"--output-file={output_file}")
            else:
                # Default output file
                enrichment_cmd.append(f"--output-file={DATA_DIR}/enriched/enriched_data.json")

            # Handle advanced options
            if "enable_entity_extraction" in parameters:
                enrichment_cmd.append(f"--enable-entity-extraction={str(parameters['enable_entity_extraction']).lower()}")

            if "enable_summarization" in parameters:
                enrichment_cmd.append(f"--enable-summarization={str(parameters['enable_summarization']).lower()}")

            if "enable_keyword_extraction" in parameters:
                enrichment_cmd.append(f"--enable-keyword-extraction={str(parameters['enable_keyword_extraction']).lower()}")

            if "use_gpu" in parameters:
                enrichment_cmd.append(f"--use-gpu={str(parameters['use_gpu']).lower()}")

            # Execute enrichment
            logger.info(f"Executing enrichment step: {' '.join(enrichment_cmd)}")
            process_logs[script_id].append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting enrichment step...")

            enrichment_process = subprocess.Popen(
                enrichment_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Read and log output
            for line in iter(enrichment_process.stdout.readline, ''):
                if not line:
                    break
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"[{timestamp}] [ENRICHMENT] {line.strip()}"
                process_logs[script_id].append(log_entry)

                # Broadcast log to connected clients
                if script_id in connected_clients:
                    for websocket in connected_clients[script_id]:
                        try:
                            await websocket.send_text(log_entry)
                        except Exception as e:
                            logger.error(f"Error sending to websocket: {e}")

            # Wait for enrichment to complete
            enrichment_return_code = enrichment_process.wait()
            if enrichment_return_code != 0:
                error_message = f"Enrichment step failed with code {enrichment_return_code}"
                logger.error(error_message)
                process_logs[script_id].append(f"[ERROR] {error_message}")
                raise ValueError(error_message)

        # Both steps completed successfully
        process_logs[script_id].append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Content extraction and enrichment completed successfully")

        # Update progress tracking
        script_progress[script_id]["progress_percent"] = 100.0
        script_progress[script_id]["status"] = "completed"
        script_progress[script_id]["end_time"] = datetime.now().isoformat()

        # Broadcast completion to connected clients
        if script_id in connected_clients:
            for websocket in connected_clients[script_id]:
                try:
                    await websocket.send_json({
                        "type": "progress",
                        "data": {
                            "script_id": script_id,
                            "progress": script_progress[script_id]
                        }
                    })
                except Exception as e:
                    logger.error(f"Error sending to websocket: {e}")

        # Remove from active processes
        if script_id in active_processes:
            del active_processes[script_id]

        # Return early since we've handled the execution directly
        return
    # Special handling for manual_extractor.py
    elif script_id == "manual_extractor":
        # Validate paths
        errors = []

        # Check if at least one source is provided
        has_url = "url" in parameters and parameters["url"]
        has_source_folder = "source_folder" in parameters and parameters["source_folder"]
        has_docker_folder = "docker_folder" in parameters and parameters["docker_folder"]

        if not has_url and not has_source_folder and not has_docker_folder:
            errors.append("Either URL, source folder, or Docker folder must be provided")

        # Validate source folder if provided
        if has_source_folder and not validate_path(parameters["source_folder"]):
            errors.append(f"Source folder does not exist: {parameters['source_folder']}")

        # Validate output directory
        output_dir = parameters.get("output_dir", f"{DATA_DIR}/extracted")
        if not os.path.exists(output_dir):
            try:
                # Try to create the output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                errors.append(f"Failed to create output directory: {output_dir} - {str(e)}")

        # If there are validation errors, raise an exception
        if errors:
            error_message = "; ".join(errors)
            logger.error(f"Validation errors: {error_message}")
            raise ValueError(error_message)

        # Handle URL parameter
        if has_url:
            cmd.append(f"--url={parameters['url']}")

        # Handle source folder parameter
        has_docker_folder = "docker_folder" in parameters and parameters["docker_folder"]

        if has_source_folder:
            source_folder = parameters['source_folder']
            # If it's a Windows path, map it to a Docker volume path
            if source_folder.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                # For now, we'll just use the path as is and let the script handle any errors
                logger.warning(f"Using Windows path in Docker container: {source_folder}")
                # Escape backslashes in Windows paths
                source_folder = source_folder.replace('\\', '\\\\')
            cmd.append(f"--local-file={source_folder}")
        elif has_docker_folder:
            docker_folder = parameters['docker_folder']
            logger.info(f"Using Docker folder path: {docker_folder}")
            cmd.append(f"--local-file={docker_folder}")

        # Handle output directory
        if "output_dir" in parameters and parameters["output_dir"]:
            cmd.append(f"--output-dir={parameters['output_dir']}")
        else:
            # Default output directory
            cmd.append(f"--output-dir={DATA_DIR}/extracted")

        # Handle extract links option
        if "extract_links" in parameters and parameters["extract_links"]:
            cmd.append("--extract-links")
    elif script_id == "data_enrichment":
        # Handle input file parameter
        if "input_file" in parameters and parameters["input_file"]:
            input_file = parameters["input_file"]
            if input_file.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                logger.warning(f"Using Windows path in Docker container: {input_file}")
                # Escape backslashes in Windows paths
                input_file = input_file.replace('\\', '\\\\')
            cmd.append(f"--input-file={input_file}")

        # Handle output file parameter
        if "output_file" in parameters and parameters["output_file"]:
            output_file = parameters["output_file"]
            if output_file.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                logger.warning(f"Using Windows path in Docker container: {output_file}")
                # Escape backslashes in Windows paths
                output_file = output_file.replace('\\', '\\\\')
            cmd.append(f"--output-file={output_file}")

        # Handle source folder parameter (from manual extractor)
        if "source_folder" in parameters and parameters["source_folder"]:
            source_folder = parameters["source_folder"]
            if source_folder.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                logger.warning(f"Using Windows path in Docker container: {source_folder}")
                # Escape backslashes in Windows paths
                source_folder = source_folder.replace('\\', '\\\\')
            cmd.append(f"--input-folder={source_folder}")
    elif script_id == "teacher_pair_generation":
        # Handle input file parameter
        if "input_file" in parameters and parameters["input_file"]:
            input_file = parameters["input_file"]
            if input_file.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                logger.warning(f"Using Windows path in Docker container: {input_file}")
                # Escape backslashes in Windows paths
                input_file = input_file.replace('\\', '\\\\')
            cmd.append(f"--input-file={input_file}")

        # Handle output file parameter
        if "output_file" in parameters and parameters["output_file"]:
            output_file = parameters["output_file"]
            if output_file.startswith(('C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                logger.warning(f"Using Windows path in Docker container: {output_file}")
                # Escape backslashes in Windows paths
                output_file = output_file.replace('\\', '\\\\')
            cmd.append(f"--output-file={output_file}")

        # Handle teacher model parameter - now maps to port
        if "teacher_model" in parameters and parameters["teacher_model"]:
            # Check if teacher_model is a container name or port
            teacher_model = parameters["teacher_model"]
            
            # Try to resolve container name to port
            container_ports = get_docker_container_ports()
            if teacher_model in container_ports:
                port = container_ports[teacher_model]
                cmd.append(f"--port={port}")
            elif teacher_model.isdigit():
                # Direct port number
                cmd.append(f"--port={teacher_model}")
            else:
                # Let the script handle it
                cmd.append(f"--port=8000")  # Default port
        
        # Add max pairs parameter if specified
        if "max_pairs" in parameters:
            cmd.append(f"--max-pairs={parameters['max_pairs']}")
    else:
        # Default parameter handling for other scripts
        for key, value in parameters.items():
            cmd.append(f"--{key}={value}")

    logger.info(f"Executing script: {' '.join(cmd)}")
    process_logs[script_id] = []

    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    active_processes[script_id] = process

    # Initialize watchdog tracking for this script
    watchdog.reset_script_tracking(script_id)
    watchdog.update_activity(script_id)

    # Read and broadcast output
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Keep the original formatting including whitespace for better readability
        log_entry = f"[{timestamp}] {line.rstrip()}"
        process_logs[script_id].append(log_entry)

        # Log to console for debugging
        logger.info(f"Script output: {log_entry}")

        # Parse progress information from the log line
        progress_info = parse_progress_from_log(script_id, line)
        if progress_info:
            update_script_progress(script_id, progress_info)
            # Also broadcast progress update via WebSocket
            if script_id in connected_clients:
                progress_update = {
                    "type": "progress",
                    "data": {
                        "script_id": script_id,
                        "progress": script_progress[script_id]
                    }
                }
                for websocket in connected_clients[script_id]:
                    try:
                        await websocket.send_json(progress_update)
                    except Exception as e:
                        logger.error(f"Error sending progress update to websocket: {e}")

        # Broadcast log to connected clients
        if script_id in connected_clients:
            for websocket in connected_clients[script_id]:
                try:
                    await websocket.send_text(log_entry)
                except Exception as e:
                    logger.error(f"Error sending to websocket: {e}")

    # Wait for process to complete
    return_code = process.wait()
    if return_code != 0:
        error_message = f"Script exited with code {return_code}"
        logger.error(error_message)
        process_logs[script_id].append(f"[ERROR] {error_message}")

        # Update progress tracking
        script_progress[script_id]["status"] = "error"
        script_progress[script_id]["error_message"] = error_message
        script_progress[script_id]["end_time"] = datetime.now().isoformat()

        # Broadcast error to connected clients
        if script_id in connected_clients:
            for websocket in connected_clients[script_id]:
                try:
                    await websocket.send_text(f"[ERROR] {error_message}")
                    # Also send progress update
                    await websocket.send_json({
                        "type": "progress",
                        "data": {
                            "script_id": script_id,
                            "progress": script_progress[script_id]
                        }
                    })
                except Exception as e:
                    logger.error(f"Error sending to websocket: {e}")
    else:
        # Script completed successfully
        script_progress[script_id]["progress_percent"] = 100.0
        script_progress[script_id]["status"] = "completed"
        script_progress[script_id]["end_time"] = datetime.now().isoformat()

        # Broadcast completion to connected clients
        if script_id in connected_clients:
            for websocket in connected_clients[script_id]:
                try:
                    await websocket.send_json({
                        "type": "progress",
                        "data": {
                            "script_id": script_id,
                            "progress": script_progress[script_id]
                        }
                    })
                except Exception as e:
                    logger.error(f"Error sending to websocket: {e}")

    # Remove from active processes
    if script_id in active_processes:
        del active_processes[script_id]

async def get_system_status() -> SystemStatus:
    """Get current system status."""
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent

    gpu_info = None
    if ENABLE_GPU:
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = [
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_percent": gpu.memoryUtil * 100,
                    "temperature": gpu.temperature,
                    "gpu_utilization": gpu.load * 100  # Add GPU compute utilization
                }
                for gpu in gpus
            ]
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")

    return SystemStatus(
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        gpu_info=gpu_info,
        active_scripts=list(active_processes.keys())
    )

# Import the vLLM client
from .vllm_client import query_vllm_model, get_docker_container_ports

# Model Query Endpoint
@app.post("/api/model/query")
async def query_model(query_data: Dict[str, Any]):
    """Query a model with the given prompt."""
    try:
        model_path = query_data.get("model_path", "")
        prompt = query_data.get("prompt", "")
        max_tokens = query_data.get("max_tokens", 500)
        temperature = query_data.get("temperature", 0.7)

        if not model_path or not prompt:
            raise HTTPException(status_code=400, detail="Model path and prompt are required")

        # Check if the model_path is a Docker container name
        container_ports = get_docker_container_ports()

        # If model_path is a container name or contains a port number
        if model_path in container_ports:
            # Use the port for the container
            port = container_ports[model_path]
            response_text = query_vllm_model(port, prompt, "/model", max_tokens, temperature)
            return {"response": response_text}
        elif model_path.startswith("port:"):
            # Extract port number from model_path
            try:
                port = int(model_path.split(":")[1])
                response_text = query_vllm_model(port, prompt, "/model", max_tokens, temperature)
                return {"response": response_text}
            except (ValueError, IndexError):
                raise HTTPException(status_code=400, detail=f"Invalid port format: {model_path}")
        elif model_path in ["8000", "8001", "8002"]:
            # Direct port number
            port = int(model_path)
            response_text = query_vllm_model(port, prompt, "/model", max_tokens, temperature)
            return {"response": response_text}
        else:
            # Try to use the traditional method with evaluate_model.py
            # Validate model path
            if not os.path.exists(model_path) and not model_path.startswith(('/data', 'C:', 'D:', 'E:', 'F:', 'G:', 'H:')):
                # If not a valid path, try to use port 8002 (student model) as fallback
                logger.warning(f"Model path does not exist: {model_path}, trying student model on port 8002")
                response_text = query_vllm_model(8002, prompt, "/model", max_tokens, temperature)
                return {"response": response_text}

            # Prepare command to query the model
            cmd = ["python3", os.path.join(SCRIPTS_DIR, "evaluate_model.py"),
                f"--model-path={model_path}",
                f"--prompt={prompt}"]

            # Execute the command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Get the output
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"Error querying model: {stderr}")
                # Try fallback to vLLM on port 8002
                logger.warning("Trying fallback to student model on port 8002")
                response_text = query_vllm_model(8002, prompt, "/model", max_tokens, temperature)
                return {"response": response_text}

            # Parse the output to extract the model's response
            response = stdout.strip()

            return {"response": response}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying model: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying model: {str(e)}")

# Docker Container Endpoints
@app.get("/api/docker/containers")
async def list_docker_containers():
    """List all available Docker containers."""
    try:
        # Check if Docker CLI is available
        try:
            # Get list of running containers
            result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}\t{{.Image}}\t{{.Ports}}'],
                                  capture_output=True, text=True, check=True)

            containers = []
            container_ports = get_docker_container_ports()

            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[0]
                        image = parts[1]
                        ports = parts[2] if len(parts) > 2 else ""

                        # Determine container type based on name and port
                        container_type = "Unknown"
                        port = None

                        if name in container_ports:
                            port = container_ports[name]
                            
                            # Check against configured model ports
                            for model_type, config in MODEL_CONFIG.items():
                                if port == config['port']:
                                    container_type = f"{config['name']} Model"
                                    break
                            
                            # If not found in config, use generic naming
                            if container_type == "Unknown":
                                if port == 8000:
                                    container_type = "Model on port 8000"
                                elif port == 8001:
                                    container_type = "Model on port 8001"
                                elif port == 8002:
                                    container_type = "Model on port 8002"

                        # Determine model type from name if not already set
                        if container_type == "Unknown":
                            if "teacher" in name.lower():
                                container_type = "Teacher Model"
                            elif "student" in name.lower():
                                container_type = "Student Model"
                            elif "vllm" in name.lower():
                                container_type = "vLLM Model"

                        container_info = {
                            "name": name,
                            "image": image,
                            "status": "running",
                            "type": container_type,
                            "port": port
                        }

                        containers.append(container_info)

            # Get list of all containers (including stopped ones)
            result = subprocess.run(['docker', 'ps', '-a', '--format', '{{.Names}}\t{{.Image}}\t{{.Status}}'],
                                  capture_output=True, text=True, check=True)

            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        name, image, status = parts[0], parts[1], parts[2]
                        # Check if this container is already in the list (running)
                        if not any(c["name"] == name for c in containers):
                            containers.append({
                                "name": name,
                                "image": image,
                                "status": "stopped" if "Exited" in status else status
                            })

            # Filter for various models
            model_containers = []

            # Define model patterns to look for
            model_patterns = [
                {"pattern": ["phi4", "phi-4", "phi 4"], "type": "Teacher Model (Phi-4)"},
                {"pattern": ["phi3", "phi-3", "phi 3"], "type": "Student Model (Phi-3)"},
                {"pattern": ["phi2", "phi-2", "phi 2"], "type": "Student Model (Phi-2)"},
                {"pattern": ["llama", "llama2", "llama-2"], "type": "LLaMA Model"},
                {"pattern": ["mistral"], "type": "Mistral Model"},
                {"pattern": ["deepseek"], "type": "DeepSeek Model"}
            ]

            # Look for containers matching the patterns
            for container in containers:
                container_name = container["name"].lower()
                container_image = container["image"].lower()

                # Check if container matches any model pattern
                for pattern_info in model_patterns:
                    patterns = pattern_info["pattern"]
                    model_type = pattern_info["type"]

                    if any(pattern in container_name or pattern in container_image for pattern in patterns):
                        model_containers.append({
                            "name": container["name"],
                            "image": container["image"],
                            "status": container["status"],
                            "type": model_type
                        })
                        break

            # If no specific models found, look for any vLLM containers
            if not model_containers:
                vllm_containers = [c for c in containers if "vllm" in c["name"].lower() or
                                                         "vllm" in c["image"].lower() or
                                                         "phi" in c["name"].lower() or
                                                         "phi" in c["image"].lower()]
                for container in vllm_containers:
                    model_containers.append({
                        "name": container["name"],
                        "image": container["image"],
                        "status": container["status"],
                        "type": "AI Model"
                    })

            return model_containers
        except FileNotFoundError:
            # Docker CLI not available, return default models
            logger.warning("Docker CLI not available, returning default models")
            return [
                {
                    "name": "phi4_gptq_vllm",
                    "image": "vllm/vllm-openai:latest",
                    "status": "running",
                    "type": "Teacher Model (Phi-4)",
                    "port": 8000
                },
                {
                    "name": "phi3_gptq_vllm",
                    "image": "vllm/vllm-openai:latest",
                    "status": "running",
                    "type": "Student Model (Phi-3)",
                    "port": 8002
                }
            ]
    except Exception as e:
        logger.error(f"Error listing Docker containers: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing Docker containers: {str(e)}")

# Model availability check endpoint
@app.get("/api/models/check-availability")
async def check_model_availability():
    """Check availability of all model servers and provide diagnostic information"""
    from .vllm_client import detect_all_vllm_servers
    
    try:
        # Detect all available servers
        servers = detect_all_vllm_servers(check_ssh_ports=True)
        
        # Process servers to add better display names and types based on environment configuration
        for server in servers:
            port = server.get('port', 0)
            
            # Check against configured model ports
            model_found = False
            for model_type, config in MODEL_CONFIG.items():
                if port == config['port']:
                    server['name'] = f"{config['name']} Model"
                    server['type'] = f"{config['name']} Model"
                    server['model_id'] = config['id']
                    server['configured_as'] = model_type
                    model_found = True
                    break
            
            # If not found in configuration, use generic naming
            if not model_found:
                container_name = server['name'].lower()
                if 'teacher' in container_name:
                    server['type'] = 'Teacher Model'
                elif 'student' in container_name:
                    server['type'] = 'Student Model'
                else:
                    server['type'] = f'Model on port {port}'
        
        # Categorize servers
        docker_servers = [s for s in servers if s.get('source_type') == 'docker']
        ssh_servers = [s for s in servers if s.get('source_type') == 'ssh']
        active_servers = [s for s in servers if s['status'] == 'active']
        
        # Create diagnostic response
        response = {
            'summary': {
                'total_servers': len(servers),
                'active_servers': len(active_servers),
                'docker_containers': len(docker_servers),
                'ssh_forwarded': len(ssh_servers)
            },
            'servers': servers,
            'recommendations': [],
            'model_config': MODEL_CONFIG  # Include configuration for debugging
        }
        
        # Add recommendations based on findings
        if len(active_servers) == 0:
            response['recommendations'].append({
                'type': 'error',
                'message': 'No active vLLM servers found. Please start Docker containers or set up SSH port forwarding.'
            })
        elif len(active_servers) < 2:
            response['recommendations'].append({
                'type': 'warning',
                'message': f'Only {len(active_servers)} server(s) active. Consider starting more for teacher/student model pairs.'
            })
        
        # Check for configured models
        teacher_port = MODEL_CONFIG['teacher']['port']
        student_port = MODEL_CONFIG['student']['port']
        
        teacher_found = any(s['port'] == teacher_port and s['status'] == 'active' for s in servers)
        student_found = any(s['port'] == student_port and s['status'] == 'active' for s in servers)
        
        if not teacher_found:
            response['recommendations'].append({
                'type': 'warning',
                'message': f'Teacher model not found on configured port {teacher_port}. Check your environment configuration.'
            })
        
        if not student_found:
            response['recommendations'].append({
                'type': 'info',
                'message': f'Student model not found on configured port {student_port}. This is normal if not training.'
            })
        
        return response
        
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking model availability: {str(e)}")

# vLLM Models Endpoint
@app.get("/api/vllm/models")
async def list_vllm_models():
    """List all available vLLM models."""
    try:
        # Get Docker container ports
        container_ports = get_docker_container_ports()

        # Check each port for available models
        models = []

        # Standard ports to check
        ports_to_check = [8000, 8001, 8002]

        # Add any additional ports from containers
        for port in container_ports.values():
            if port not in ports_to_check:
                ports_to_check.append(port)

        # Check each port
        from .vllm_client import get_vllm_client_for_port

        for port in ports_to_check:
            try:
                client = get_vllm_client_for_port(port)
                available_models = client.get_available_models()

                # Determine model type based on configured ports
                model_type = "Unknown"
                for mt, config in MODEL_CONFIG.items():
                    if port == config['port']:
                        model_type = f"{config['name']} Model"
                        break

                for model_data in available_models:
                    model_id = model_data.get("id", "Unknown")
                    model_name = model_id

                    # Try to get a more user-friendly name
                    if "/" in model_id:
                        model_name = model_id.split("/")[-1]

                    models.append({
                        "name": model_name,
                        "path": f"port:{port}",  # Use port as the path
                        "description": f"vLLM model on port {port}",
                        "type": model_type,
                        "port": port,
                        "model_id": model_id
                    })
            except Exception as e:
                logger.warning(f"Could not connect to vLLM server on port {port}: {e}")

        # If no models were found, add default models
        if not models:
            models = [
                {
                    "name": "Phi-4 Teacher",
                    "path": "port:8000",
                    "description": "Phi-4 teacher model on port 8000",
                    "type": "Teacher Model",
                    "port": 8000,
                    "model_id": "/model"
                },
                {
                    "name": "Phi-3 Student",
                    "path": "port:8002",
                    "description": "Phi-3 student model on port 8002",
                    "type": "Student Model",
                    "port": 8002,
                    "model_id": "/model"
                }
            ]

        return models
    except Exception as e:
        logger.error(f"Error listing vLLM models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing vLLM models: {str(e)}")

# API Endpoints
@app.get("/api/scripts")
async def list_scripts():
    """List all available scripts."""
    scripts = [
        {
            "id": "manual_extractor",
            "name": "Manual Extractor",
            "description": "Extract content from technical documents to PDF and JSON",
            "step": 1,
            "hidden": True  # Hide from pipeline view
        },
        {
            "id": "data_enrichment",
            "name": "Data Enrichment",
            "description": "Clean text, extract entities, and generate summaries using GPU acceleration",
            "step": 2,
            "hidden": True  # Hide from pipeline view
        },
        {
            "id": "content_extraction_enrichment",
            "name": "Content Extraction & Enrichment",
            "description": "Extract and enrich content from various sources in a single step",
            "step": 1
        },
        {
            "id": "teacher_pair_generation",
            "name": "Teacher Pair Generation",
            "description": "Generate teacher model outputs using Phi-4 via vLLM",
            "step": 3
        },
        {
            "id": "distillation",
            "name": "Distillation Training",
            "description": "Train student model using enhanced prompt engineering and parameters",
            "step": 4
        },
        {
            "id": "student_self_study",
            "name": "Student Self-Study",
            "description": "Enable the distilled model to learn from additional domain-specific content",
            "step": 5
        },
        {
            "id": "merge_model",
            "name": "Model Merging",
            "description": "Merge the base model with trained LoRA adapters for deployment",
            "step": 6
        },
        {
            "id": "evaluation",
            "name": "Model Evaluation",
            "description": "Test distilled model on sample prompts",
            "step": 7
        }
    ]
    return scripts

@app.get("/api/scripts/{script_id}/config")
async def get_script_config(script_id: str):
    """Get configuration template for a script."""
    config_templates = {
        "manual_extractor": {
            "url": "https://example.com/docs",
            "source_folder": "",
            "docker_folder": "/data",
            "output_dir": f"{DATA_DIR}/extracted"
        },
        "data_enrichment": {
            "input_file": f"{DATA_DIR}/extracted/extracted_data.json",
            "output_file": f"{DATA_DIR}/enriched/enriched_data.json",
            "source_folder": ""
        },
        "content_extraction_enrichment": {
            "url": "https://example.com/docs",
            "source_folder": "",
            "docker_folder": "/data",
            "output_dir": f"{DATA_DIR}/extracted",
            "extract_links": False,
            "enable_enrichment": True,
            "input_file": f"{DATA_DIR}/extracted/extracted_data.json",
            "output_file": f"{DATA_DIR}/enriched/enriched_data.json",
            "enable_entity_extraction": True,
            "enable_summarization": True,
            "enable_keyword_extraction": True,
            "use_gpu": True
        },
        "teacher_pair_generation": {
            "input_file": f"{DATA_DIR}/enriched/enriched_data.json",
            "output_file": f"{DATA_DIR}/teacher_pairs/teacher_pairs.json",
            "teacher_model": "jakiAJK/microsoft-phi-4_GPTQ-int4"
        },
        "distillation": {
            "input_file": f"{DATA_DIR}/teacher_pairs/teacher_pairs.json",
            "output_dir": f"{DATA_DIR}/distilled_model",
            "teacher_model": "phi4_gptq_vllm",
            "student_model": "phi3_gptq_vllm",
            "num_epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "gradient_accumulation_steps": 8,
            "beta": 0.1,
            "lambda": 0.1,
            "max_seq_length": 256,
            "use_gpu": True,
            "mixed_precision": True,
            "use_4bit": True
        },
        "student_self_study": {
            "pdf_folder": f"{DATA_DIR}/domain_pdfs",
            "model_path": f"{DATA_DIR}/distilled_model/best_checkpoint",
            "output_dir": f"{DATA_DIR}/self_study_results",
            "num_questions": 3,
            "use_hierarchical_context": True,
            "include_reasoning": True
        },
        "merge_model": {
            "adapter_path": f"{DATA_DIR}/distilled_model/best_checkpoint",
            "output_path": f"{DATA_DIR}/merged_distilled_model"
        },
        "evaluation": {
            "model_path": f"{DATA_DIR}/merged_distilled_model",
            "test_prompt": "Extract detailed technical requirements for a new IoT device in the healthcare domain:"
        }
    }

    if script_id not in config_templates:
        raise HTTPException(status_code=404, detail=f"Script '{script_id}' not found")

    return config_templates[script_id]

@app.post("/api/scripts/{script_id}/execute")
async def execute_script(script_id: str, config: Dict[str, Any]):
    """Execute a script with the given configuration."""
    if script_id in active_processes:
        raise HTTPException(status_code=400, detail=f"Script '{script_id}' is already running")

    try:
        # Start script execution in background task
        asyncio.create_task(run_script(script_id, config))
        return {"message": f"Script '{script_id}' started successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing script: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing script: {str(e)}")

@app.get("/api/scripts/{script_id}/status")
async def get_script_status(script_id: str):
    """Get detailed status for a specific script."""
    # Check if script is currently running
    is_running = script_id in active_processes

    # Get progress information if available
    progress = script_progress.get(script_id, {
        "progress_percent": 0.0,
        "current_step": None,
        "steps_completed": 0,
        "total_steps": 0,
        "memory_usage": None
    })

    # Determine status
    if is_running:
        status = "running"
    elif script_id in script_progress and script_progress[script_id].get("status") == "error":
        status = "error"
    elif script_id in script_progress and script_progress[script_id].get("status") == "completed":
        status = "completed"
    else:
        status = "pending"

    # Check if script is stuck according to watchdog
    is_stuck = script_id in watchdog.stuck_processes

    # Create response
    response = {
        "script_id": script_id,
        "status": status,
        "is_running": is_running,
        "is_stuck": is_stuck,
        "progress_percent": progress.get("progress_percent", 0.0),
        "current_step": progress.get("current_step"),
        "steps_completed": progress.get("steps_completed", 0),
        "total_steps": progress.get("total_steps", 0),
        "start_time": progress.get("start_time"),
        "end_time": progress.get("end_time"),
        "error_message": progress.get("error_message"),
        "memory_usage": progress.get("memory_usage")
    }

    # If the script is stuck, add stuck info
    if is_stuck:
        response["stuck_info"] = watchdog.stuck_processes[script_id]

    return response

@app.get("/api/scripts/{script_id}/logs")
async def get_script_logs(script_id: str, limit: int = 100):
    """Get logs for a specific script."""
    logger.info(f"Getting logs for script: {script_id}, limit: {limit}")

    # Check if logs exist for this script
    if script_id not in process_logs:
        logger.warning(f"No logs found for script: {script_id}")
        # For content_extraction_enrichment, check if logs exist for manual_extractor or data_enrichment
        if script_id == "content_extraction_enrichment":
            # Initialize logs for content_extraction_enrichment if they don't exist
            process_logs[script_id] = []

            # Add a message indicating that logs are being initialized
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            process_logs[script_id].append(f"[{timestamp}] [INFO] Logs initialized for Content Extraction & Enrichment")

            logger.info(f"Initialized logs for script: {script_id}")
        else:
            raise HTTPException(status_code=404, detail=f"No logs found for script '{script_id}'")

    # Return the most recent logs up to the limit
    logs = process_logs[script_id][-limit:] if limit > 0 else process_logs[script_id]
    logger.info(f"Returning {len(logs)} logs for script: {script_id}")
    return {"logs": logs}

@app.get("/api/system/status")
async def get_system_info():
    """Get system information including CPU, memory, and GPU usage."""
    return await get_system_status()

# WebSocket endpoints
@app.websocket("/ws/scripts/{script_id}/logs")
async def websocket_script_logs(websocket: WebSocket, script_id: str):
    """WebSocket endpoint for real-time script logs."""
    logger.info(f"WebSocket connection request for script logs: {script_id}")
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for script logs: {script_id}")

    # Add client to connected clients
    if script_id not in connected_clients:
        connected_clients[script_id] = []
    connected_clients[script_id].append(websocket)
    logger.info(f"Client added to connected clients for script: {script_id}, total clients: {len(connected_clients[script_id])}")

    # For content_extraction_enrichment, ensure logs are initialized
    if script_id == "content_extraction_enrichment" and script_id not in process_logs:
        process_logs[script_id] = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        process_logs[script_id].append(f"[{timestamp}] [INFO] WebSocket connection established for Content Extraction & Enrichment")
        logger.info(f"Initialized logs for script: {script_id} via WebSocket connection")

    try:
        # Send initial logs
        if script_id in process_logs:
            logger.info(f"Sending {len(process_logs[script_id])} initial logs for script: {script_id}")
            for log in process_logs[script_id]:
                await websocket.send_text(log)
        else:
            logger.warning(f"No logs found for script: {script_id} to send via WebSocket")

        # Send initial progress if available
        if script_id in script_progress:
            logger.info(f"Sending initial progress for script: {script_id}")
            await websocket.send_json({
                "type": "progress",
                "data": {
                    "script_id": script_id,
                    "progress": script_progress[script_id]
                }
            })

        # Keep connection open
        while True:
            # Wait for any message from client (ping)
            data = await websocket.receive_text()
            # Echo back as pong
            await websocket.send_text(f"pong: {data}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from script {script_id}")
    finally:
        # Remove client from connected clients
        if script_id in connected_clients and websocket in connected_clients[script_id]:
            connected_clients[script_id].remove(websocket)
            logger.info(f"Client removed from connected clients for script: {script_id}, remaining clients: {len(connected_clients[script_id])}")

@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for real-time system status updates."""
    await websocket.accept()

    try:
        # Send initial status
        status = await get_system_status()
        await websocket.send_json({
            "type": "status",
            "data": {
                "cpu_percent": status.cpu_percent,
                "memory_percent": status.memory_percent,
                "gpu_info": status.gpu_info,
                "active_scripts": status.active_scripts,
                "watchdog_status": watchdog.get_status()
            }
        })

        # Keep connection open and send updates periodically
        while True:
            # Wait for a ping or send updates every 5 seconds
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                # Echo back as pong
                await websocket.send_text(f"pong: {data}")
            except asyncio.TimeoutError:
                # Send status update
                status = await get_system_status()
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "cpu_percent": status.cpu_percent,
                        "memory_percent": status.memory_percent,
                        "gpu_info": status.gpu_info,
                        "active_scripts": status.active_scripts,
                        "watchdog_status": watchdog.get_status()
                    }
                })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from status endpoint")

@app.websocket("/ws/resources")
async def websocket_resources(websocket: WebSocket):
    """WebSocket endpoint for real-time resource monitoring."""
    await websocket.accept()

    try:
        # Keep connection open and send updates periodically
        while True:
            # Wait for a ping or send updates every 2 seconds
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
                # Echo back as pong
                await websocket.send_text(f"pong: {data}")
            except asyncio.TimeoutError:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent

                gpu_info = None
                if ENABLE_GPU:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            gpu_info = {
                                "memory_percent": gpu.memoryUtil * 100,
                                "utilization": gpu.load * 100,
                                "temperature": gpu.temperature
                            }
                    except Exception as e:
                        logger.error(f"Error getting GPU info: {e}")

                # Send resource update
                await websocket.send_json({
                    "type": "resources",
                    "data": {
                        "timestamp": datetime.now().isoformat(),
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "gpu_info": gpu_info
                    }
                })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from resources endpoint")

# Watchdog endpoints
@app.get("/api/watchdog/status")
async def get_watchdog_status():
    """Get the current status of the watchdog."""
    return watchdog.get_status()

@app.post("/api/watchdog/action")
async def perform_watchdog_action(action_data: WatchdogAction):
    """Perform a watchdog action."""
    if not action_data.script_id:
        raise HTTPException(status_code=400, detail="Script ID is required")

    try:
        result = watchdog.perform_action(action_data.action, action_data.script_id)
        return result
    except Exception as e:
        logger.error(f"Error performing watchdog action: {e}")
        raise HTTPException(status_code=500, detail=f"Error performing watchdog action: {str(e)}")

# Novelty insights endpoints
@app.get("/api/novelty/data")
async def get_novelty_data(max_lines: int = 20, limit: int = 10):
    """Get all novelty data for the dashboard."""
    # Define paths for log files
    curiosity_log_path = os.environ.get("CURIOSITY_LOG_PATH", "curiosity.log")
    novelty_counts_path = os.environ.get("NOVELTY_COUNTS_PATH", "novelty_counts.json")
    learning_log_path = os.environ.get("LEARNING_LOG_PATH", "learning_log_metasploit.json")

    # Initialize response data
    response_data = {
        'timeline': [],
        'most_novel': [],
        'least_novel': [],
        'statistics': {
            "total_states": 0,
            "total_visits": 0,
            "avg_visits_per_state": 0,
            "max_visits": 0,
            "min_visits": 0,
            "unique_states": 0
        },
        'recent_logs': []
    }

    # Read curiosity log
    try:
        if os.path.exists(curiosity_log_path):
            with open(curiosity_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                response_data['recent_logs'] = [line.strip() for line in lines[-max_lines:]]
    except Exception as e:
        logger.error(f"Error reading curiosity log: {e}")

    # Read novelty counts
    try:
        if os.path.exists(novelty_counts_path):
            with open(novelty_counts_path, 'r', encoding='utf-8') as f:
                novelty_counts = json.load(f)

                # Calculate statistics
                total_states = len(novelty_counts)
                total_visits = sum(novelty_counts.values())
                avg_visits = total_visits / total_states if total_states > 0 else 0
                max_visits = max(novelty_counts.values()) if novelty_counts else 0
                min_visits = min(novelty_counts.values()) if novelty_counts else 0
                unique_states = sum(1 for count in novelty_counts.values() if count == 1)

                response_data['statistics'] = {
                    "total_states": total_states,
                    "total_visits": total_visits,
                    "avg_visits_per_state": avg_visits,
                    "max_visits": max_visits,
                    "min_visits": min_visits,
                    "unique_states": unique_states
                }

                # Get most novel states
                state_counts = [(state, count) for state, count in novelty_counts.items()]
                state_counts.sort(key=lambda x: x[1])  # Sort by count (ascending)
                response_data['most_novel'] = [
                    {'state': state, 'count': count, 'novelty_score': 1.0 / (count ** 0.5)}
                    for state, count in state_counts[:limit]
                ]

                # Get least novel states
                state_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count (descending)
                response_data['least_novel'] = [
                    {'state': state, 'count': count, 'novelty_score': 1.0 / (count ** 0.5)}
                    for state, count in state_counts[:limit]
                ]
    except Exception as e:
        logger.error(f"Error reading novelty counts: {e}")

    # Read learning log
    try:
        if os.path.exists(learning_log_path):
            with open(learning_log_path, 'r', encoding='utf-8') as f:
                learning_log = json.load(f)

                # Extract timeline data
                timeline = []
                for entry in learning_log:
                    if 'timestamp' in entry and 'novelty_score' in entry:
                        timeline.append({
                            'timestamp': entry['timestamp'],
                            'novelty_score': entry['novelty_score'],
                            'command': entry.get('command', ''),
                            'success': entry.get('success', True)
                        })

                response_data['timeline'] = timeline
    except Exception as e:
        logger.error(f"Error reading learning log: {e}")

    return response_data

@app.get("/api/novelty/logs")
async def get_novelty_logs(max_lines: int = 100):
    """Get the most recent curiosity log entries."""
    curiosity_log_path = os.environ.get("CURIOSITY_LOG_PATH", "curiosity.log")

    try:
        if os.path.exists(curiosity_log_path):
            with open(curiosity_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-max_lines:]]
        else:
            return []
    except Exception as e:
        logger.error(f"Error reading curiosity log: {e}")
        return []

# Memory monitoring endpoints
@app.get("/api/scripts/{script_id}/memory")
async def get_script_memory(script_id: str):
    """Get memory usage history for a script."""
    # This would typically come from a database or in-memory storage
    # For now, we'll return a simple mock response
    if script_id not in script_progress:
        raise HTTPException(status_code=404, detail=f"No memory data found for script '{script_id}'")

    # Get current GPU info
    gpu_info = None
    gpu_memory_percent = None
    gpu_utilization = None

    if ENABLE_GPU:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_memory_percent = gpu.memoryUtil * 100
                gpu_utilization = gpu.load * 100
                gpu_memory_used_mb = gpu.memoryUsed
        except Exception as e:
            logger.error(f"Error getting GPU info: {e}")

    # Create a mock memory history (in a real app, this would come from stored data)
    # In a production app, you would store memory snapshots over time
    current_memory = script_progress[script_id].get("memory_usage", {})

    # For now, return current memory info and some mock historical data
    memory_data = [
        {
            "timestamp": datetime.now().isoformat(),
            "gpu_memory_percent": gpu_memory_percent,
            "gpu_memory_used_mb": current_memory.get("gpu_mb", 0),
            "gpu_utilization": gpu_utilization,
            "cpu_memory_percent": psutil.virtual_memory().percent,
            "step": script_progress[script_id].get("current_step", "Unknown")
        }
    ]

    return memory_data

# Startup event to initialize the watchdog service
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Register recovery callbacks
    watchdog.register_recovery_callback("restart", lambda script_id: logger.info(f"Restart requested for {script_id}"))

    # Start the watchdog service
    asyncio.create_task(run_watchdog_service(watchdog))
    logger.info("Watchdog service started")

# Shutdown event to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    # Stop the watchdog
    watchdog.stop_monitoring()
    logger.info("Watchdog service stopped")

    # Terminate any running processes
    for script_id, process in list(active_processes.items()):
        logger.info(f"Terminating process for script {script_id}")
        try:
            process.terminate()
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
