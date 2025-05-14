import sys
import os

# Add the project root to the path to import diagnose_vllm_servers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from diagnose_vllm_servers import check_vllm_server

SERVER_CONFIG = {
    'teacher': {
        'name': 'Phi-4 Teacher Model',
        'port': 8000,
        'container_name': 'phi4_gptq_vllm'
    },
    'student': {
        'name': 'Student Model (Phi-2)',
        'port': 8002,
        'container_name': 'phi2_vllm'
    }
}

def check_server_status():
    """Check if required vLLM servers are running."""
    results = {}
    for server_type, config in SERVER_CONFIG.items():
        status = check_vllm_server("localhost", config['port'])
        results[server_type] = {
            'running': status['is_vllm'],
            'port': config['port'],
            'name': config['name'],
            'container': config['container_name']
        }
    return results
