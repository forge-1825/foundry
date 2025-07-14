from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Response
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import subprocess
import os
import socket
import logging
import json
from pathlib import Path

# Create router
pipeline_bp = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Process manager
class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.next_id = 1

    def add_process(self, process, step_id):
        process_id = str(self.next_id)
        self.next_id += 1
        self.processes[process_id] = {
            'process': process,
            'step_id': step_id,
            'status': 'running',
            'output': [],
            'error': []
        }
        return process_id

    def get_process_status(self, process_id):
        if process_id not in self.processes:
            return None

        process_info = self.processes[process_id]
        process = process_info['process']

        # Check if process has completed
        if process.poll() is not None:
            process_info['status'] = 'completed' if process.returncode == 0 else 'error'

        # Read any new output
        while True:
            line = process.stdout.readline()
            if not line:
                break
            process_info['output'].append(line.strip())

        # Read any new errors
        while True:
            line = process.stderr.readline()
            if not line:
                break
            process_info['error'].append(line.strip())

        return {
            'status': process_info['status'],
            'step_id': process_info['step_id'],
            'output': process_info['output'],
            'error': process_info['error'],
            'returncode': process.returncode if process.poll() is not None else None
        }

    def stop_process(self, process_id):
        if process_id not in self.processes:
            return False

        process = self.processes[process_id]['process']
        if process.poll() is None:  # Process is still running
            process.terminate()
            self.processes[process_id]['status'] = 'stopped'

        return True

process_manager = ProcessManager()

# Models
class PipelineStep(BaseModel):
    id: str
    name: str
    description: str
    script: str
    next_step: Optional[str] = None
    previous_step: Optional[str] = None
    parameters: Dict[str, Any] = {}
    status: str = "pending"  # pending, running, completed, error
    requires_server: Optional[List[str]] = None

class PipelineStatus(BaseModel):
    steps: List[PipelineStep]
    current_step: Optional[str] = None

class ProcessRequest(BaseModel):
    step_id: str
    params: Dict[str, Any] = {}

# Server configuration
SERVER_CONFIG = {
    'teacher': {
        'name': 'Llama 3 8B Instruct',
        'port': 8000,
        'type': 'teacher'
    },
    'student': {
        'name': 'Phi-3 Mini',
        'port': 8002,
        'type': 'student'
    },
    'distilled': {
        'name': 'Phi-3 Mini Distilled',
        'port': 8003,
        'type': 'distilled'
    }
}

# Pipeline configuration
PIPELINE_CONFIG = {
    'steps': [
        {
            'id': 'post_enrichment_pipeline_selector',
            'name': 'Pipeline Selector',
            'description': 'Choose between Standard and PRD pipelines',
            'script': 'run_post_enrichment_pipeline_selector.bat',
            'next_step': 'teacher_pair_generation',
            'previous_step': 'content_extraction_enrichment',
            'parameters': {
                'pipeline_type': 'standard',
                'output_dir': 'Output',
                'prd_phase': '1',
                'prd_model_path': 'distilled_model_prd_phase3'
            }
        },
        {
            'id': 'content_extraction_enrichment',
            'name': 'Content Extraction & Enrichment',
            'description': 'Extract and enrich content from PDF files, websites, and code datasets',
            'script': 'run_data_enrichment.bat',
            'next_step': 'post_enrichment_pipeline_selector',
            'previous_step': None,
            'parameters': {
                'source_folder': '',
                'output_dir': 'Output',
                'enable_enrichment': True,
                'topic': 'cybersecurity',
                'enable_entity_extraction': True,
                'enable_summarization': True,
                'enable_keyword_extraction': True,
                'use_gpu': True,
                'process_code_dataset': True,
                'code_dataset': 'Shuu12121/python-codesearch-dataset-open',
                'code_dataset_split': 'train',
                'code_dataset_cache_dir': 'datasets_cache',
                'code_dataset_max_samples': 5000,
                'process_cve_data': True,
                'cve_data_folder': 'cvelistV5-main',
                'use_domain_context': False,
                'domain_context_file': ''
            }
        },
        {
            'id': 'teacher_pair_generation',
            'name': 'Teacher Pair Generation',
            'description': 'Generate teaching examples with hierarchical context',
            'script': 'teacher_pair_generation_vllm_hierarchical.py',
            'next_step': 'distillation',
            'previous_step': 'content_extraction_enrichment',
            'parameters': {
                'input_file': 'Output/enriched_data.json',
                'output_file': 'Output/teacher_pairs.json',
                'model': 'llama3_teacher_vllm',
                'num_pairs': 100
            },
            'requires_server': ['teacher']
        },
        {
            'id': 'distillation',
            'name': 'Distillation Training',
            'description': 'Train the student model using the teaching examples',
            'script': 'distillation_vllm_faster_improved.py',
            'next_step': 'merge_model',
            'previous_step': 'teacher_pair_generation',
            'parameters': {
                'input_file': 'Output/teacher_pairs.json',
                'output_dir': 'Output/distilled',
                'model': 'phi3_vllm',
                'base_model': 'microsoft/Phi-3-mini-4k-instruct',
                'epochs': 3,
                'batch_size': 4,
                'learning_rate': 1e-4,
                'use_peft': True,
                'lora_r': 16,
                'lora_alpha': 32
            }
        },
        {
            'id': 'merge_model',
            'name': 'Model Merging',
            'description': 'Merge the trained adapters with the base model',
            'script': 'merge_model.py',
            'next_step': 'student_self_study',
            'previous_step': 'distillation',
            'parameters': {
                'input_dir': 'Output/distilled',
                'output_dir': 'Output/merged',
                'base_model': 'microsoft/Phi-3-mini-4k-instruct',
                'model_type': 'phi-3-mini',
                'use_safetensors': True
            }
        },
        {
            'id': 'student_self_study',
            'name': 'Student Self-Study',
            'description': 'Allow the student model to further learn from the data',
            'script': 'student_self_study_enhanced.py',
            'next_step': 'evaluation',
            'previous_step': 'merge_model',
            'parameters': {
                'pdf_folder': 'AgentGreen',
                'model_path': 'Output/merged',
                'output_dir': 'Output/self_study_results',
                'use_teacher': True,
                'teacher_model': 'llama3_teacher_vllm',
                'teacher_port': 8000,
                'topics_of_interest': 'cybersecurity, network scanning, vulnerability assessment',
                'num_questions': 20,
                'iterative_refinement': True,
                'use_hierarchical_context': True,
                'include_reasoning': True,
                'use_rag': True,
                'use_taboutt': True,
                'min_sentence_length': 5,
                'max_sentence_length': 100,
                'max_paragraph_size': 5,
                'max_sentences': 10,
                'verbose': True,
                'show_thoughts': False,
                'use_8bit': False
            },
            'requires_server': ['teacher']
        },
        {
            'id': 'evaluation',
            'name': 'Model Evaluation',
            'description': 'Evaluate the performance of the distilled model',
            'script': 'run_openevals_langchain.py',
            'next_step': None,
            'previous_step': 'student_self_study',
            'parameters': {
                'config': 'eval_config_full_comparison.yaml',
                'models': ['teacher', 'student', 'distilled'],
                'datasets': ['simple_qa', 'command_extraction', 'error_suggestion'],
                'evaluators': ['string_match', 'embedding_similarity', 'qa_correctness'],
                'include_rag': True,
                'max_examples': 50
            },
            'requires_server': ['teacher', 'student', 'distilled']
        }
    ]
}

def check_server_running(server_type):
    """Check if a server is running based on its type."""
    if server_type not in SERVER_CONFIG:
        return False

    host = "localhost"
    port = SERVER_CONFIG[server_type]['port']

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception:
        return False

@pipeline_bp.post("/start")
async def start_pipeline(request: ProcessRequest):
    step_id = request.step_id
    params = request.params

    if step_id not in [step['id'] for step in PIPELINE_CONFIG['steps']]:
        raise HTTPException(status_code=400, detail="Invalid step ID")

    step = next(s for s in PIPELINE_CONFIG['steps'] if s['id'] == step_id)

    # Check server requirements
    if step.get('requires_server'):
        missing_servers = []
        server_details = []
        
        for server in step['requires_server']:
            if not check_server_running(server):
                missing_servers.append(server)
                port = SERVER_CONFIG.get(server, {}).get('port', 'unknown')
                server_details.append(f"{server} (port {port})")
        
        if missing_servers:
            # Import the detection function
            import sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from app.vllm_client import detect_all_vllm_servers
            
            # Detect available servers
            available_servers = detect_all_vllm_servers()
            available_info = []
            for srv in available_servers:
                if srv['status'] == 'active':
                    available_info.append(f"{srv['name']} on port {srv['port']} ({srv['type']})")
            
            error_msg = f"Required model server(s) not running: {', '.join(server_details)}. "
            
            if available_info:
                error_msg += f"Available servers: {', '.join(available_info)}. "
                error_msg += "Please update your configuration to use an available server."
            else:
                error_msg += "No vLLM servers detected. Please start Docker containers or set up SSH port forwarding."
            
            raise HTTPException(
                status_code=400,
                detail=error_msg
            )

    # Special handling for content_extraction_enrichment step
    if step_id == 'content_extraction_enrichment':
        # This step uses a batch file with positional arguments
        script_path = os.path.join('scripts', step['script'])

        # Extract parameters
        source_folder = params.get('source_folder', '')
        output_dir = params.get('output_dir', 'Output')
        process_code_dataset = str(params.get('process_code_dataset', True)).lower()
        code_dataset = params.get('code_dataset', 'Shuu12121/python-codesearch-dataset-open')
        code_dataset_split = params.get('code_dataset_split', 'train')
        code_dataset_max_samples = str(params.get('code_dataset_max_samples', 5000))
        topic = params.get('topic', 'cybersecurity')
        process_cve_data = str(params.get('process_cve_data', True)).lower()
        cve_data_folder = params.get('cve_data_folder', 'cvelistV5-main')
        use_domain_context = str(params.get('use_domain_context', False)).lower()
        domain_context_file = params.get('domain_context_file', '')

        # Build command with positional arguments
        command = [
            script_path,
            source_folder,
            output_dir,
            process_code_dataset,
            code_dataset,
            code_dataset_split,
            code_dataset_max_samples,
            topic,
            process_cve_data,
            cve_data_folder
        ]

        # Add domain context parameters if enabled
        if use_domain_context == 'true' and domain_context_file:
            command.extend(['true', domain_context_file])
        else:
            command.extend(['false', ''])

        logger.info(f"Running content extraction with command: {command}")
    else:
        # Standard parameter handling for other steps
        script_path = os.path.join('scripts', step['script'])
        command = ['python', script_path]

        # Add parameters
        for key, value in params.items():
            command.extend([f'--{key}', str(value)])

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        process_id = process_manager.add_process(process, step_id)

        return {
            'status': 'started',
            'process_id': process_id,
            'step': step_id
        }

    except Exception as e:
        logger.error(f"Error starting pipeline step {step_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_bp.get("/status/{process_id}")
async def get_pipeline_process_status(process_id: str):
    status = process_manager.get_process_status(process_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Process not found")
    return status

@pipeline_bp.post("/stop/{process_id}")
async def stop_pipeline_process(process_id: str):
    if process_manager.stop_process(process_id):
        return {'status': 'stopped'}
    raise HTTPException(status_code=404, detail="Process not found")

@pipeline_bp.get("/steps")
async def get_pipeline_steps():
    """Get all pipeline steps"""
    return PIPELINE_CONFIG['steps']

@pipeline_bp.get("/steps/{step_id}")
async def get_pipeline_step(step_id: str):
    """Get a specific pipeline step"""
    for step in PIPELINE_CONFIG['steps']:
        if step["id"] == step_id:
            return step
    raise HTTPException(status_code=404, detail=f"Step {step_id} not found")

@pipeline_bp.get("/status")
async def get_pipeline_status():
    """Get the current status of the pipeline"""
    # In a real implementation, this would read from a database or state file
    # For now, we'll just return the steps with default status
    return {"steps": PIPELINE_CONFIG['steps'], "current_step": None}

@pipeline_bp.get("/models")
async def get_available_models():
    """Get available models from Docker containers"""
    # In a real implementation, this would query Docker for running containers
    # For now, we'll return hardcoded values
    return [
        {
            "id": "llama3_teacher_vllm",
            "name": "Llama 3 8B Instruct AWQ",
            "type": "teacher",
            "status": "running",
            "port": 8001
        },
        {
            "id": "phi3_vllm",
            "name": "Phi-3 Mini",
            "type": "student",
            "status": "running",
            "port": 8002
        }
    ]
