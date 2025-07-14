import json
import os
import sys
import time
import logging
import subprocess
import argparse
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Tuple, Optional

# Configure logging to output messages with timestamps.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

class VLLMModelManager:
    """Manages vLLM model detection and selection with SSH forwarding support"""
    
    def __init__(self):
        self.available_models = []
        self.selected_model = None
        self.selected_port = None
        self.client = None
        self.use_remote_models = os.environ.get('USE_REMOTE_MODELS', 'True').lower() == 'true'
        self.is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == '1'
        
    def get_host(self):
        """Get the appropriate hostname for accessing services"""
        if self.use_remote_models and self.is_docker:
            return 'host.docker.internal'
        return 'localhost'
        
    def detect_ssh_forwarded_ports(self) -> List[Dict[str, any]]:
        """Detect SSH-forwarded vLLM ports"""
        ssh_ports = [8000, 8001, 8002, 8003]
        containers = []
        host = self.get_host()
        
        logging.info(f"Checking SSH-forwarded ports on {host}...")
        
        for port in ssh_ports:
            try:
                import requests
                response = requests.get(f"http://{host}:{port}/v1/models", timeout=2)
                if response.status_code == 200:
                    models_data = response.json()
                    if 'data' in models_data and models_data['data']:
                        model_id = models_data['data'][0].get('id', '/model')
                        model_type = self.detect_model_type("", model_id, port)
                        
                        containers.append({
                            'name': f'ssh_port_{port}',
                            'port': port,
                            'type': model_type,
                            'model_id': model_id,
                            'source': 'ssh'
                        })
                        logging.info(f"Found SSH-forwarded vLLM model on port {port}: {model_id}")
            except Exception as e:
                logging.debug(f"Port {port} not accessible: {e}")
                
        return containers
        
    def detect_running_containers(self) -> List[Dict[str, any]]:
        """Detect all running Docker containers with vLLM models"""
        containers = []
        
        # First try SSH-forwarded ports
        if self.use_remote_models:
            containers = self.detect_ssh_forwarded_ports()
            if containers:
                return containers
        
        # Then try local Docker containers
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}'],
                capture_output=True, text=True, check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 3:
                    name, image, ports = parts[0], parts[1], parts[2]
                    
                    if 'vllm' in image.lower() or 'vllm' in name.lower():
                        port_match = None
                        for port_mapping in ports.split(','):
                            if '->' in port_mapping and 'tcp' in port_mapping:
                                try:
                                    public_port = port_mapping.split('->')[0].split(':')[-1]
                                    port_match = int(public_port)
                                    break
                                except:
                                    continue
                        
                        if port_match:
                            model_type = self.detect_model_type(name, image, port_match)
                            containers.append({
                                'name': name,
                                'image': image,
                                'port': port_match,
                                'type': model_type,
                                'source': 'docker'
                            })
                            logging.info(f"Found Docker container: {name} on port {port_match}")
                            
        except subprocess.CalledProcessError as e:
            logging.warning(f"Could not run docker ps: {e}")
        except Exception as e:
            logging.error(f"Error detecting Docker containers: {e}")
            
        return containers
    
    def detect_model_type(self, name: str, image: str, port: int) -> str:
        """Detect whether a model is teacher or student based on various heuristics"""
        combined = f"{name} {image} port:{port}".lower()
        
        if port == 8000:
            return 'teacher'
        elif port in [8001, 8002, 8003]:
            return 'student'
        
        if any(teacher in combined for teacher in ['phi4', 'phi-4', 'llama', 'mixtral', 'gpt']):
            return 'teacher'
        elif any(student in combined for student in ['phi3', 'phi-3', 'phi2', 'phi-2']):
            return 'student'
        else:
            return 'unknown'
    
    def test_vllm_endpoint(self, port: int, timeout: int = 30) -> Tuple[bool, Optional[str]]:
        """Test if vLLM endpoint is responding and get model info"""
        host = self.get_host()
        logging.info(f"Testing vLLM endpoint on {host}:{port}...")
        
        try:
            client = OpenAI(
                base_url=f"http://{host}:{port}/v1",
                api_key="not-needed"
            )
            
            # Try to get model list first
            try:
                import requests
                response = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    if 'data' in models_data and models_data['data']:
                        model_id = models_data['data'][0].get('id', '/model')
                        logging.info(f"Found model: {model_id} on port {port}")
                        return True, model_id
            except Exception as e:
                logging.debug(f"Could not fetch model list: {e}")
            
            # Try a simple completion as fallback
            response = client.completions.create(
                model="/model",
                prompt="Hello",
                max_tokens=1,
                temperature=0.1
            )
            
            logging.info(f"vLLM server on port {port} is responding")
            return True, "/model"
            
        except Exception as e:
            logging.warning(f"vLLM server on {host}:{port} not responding: {e}")
            return False, None
    
    def select_best_model(self, containers: List[Dict[str, any]], prefer_teacher: bool = True) -> Optional[Dict[str, any]]:
        """Select the best available model based on type preference"""
        if not containers:
            return None
        
        # Test each container
        working_containers = []
        for container in containers:
            is_working, model_id = self.test_vllm_endpoint(container['port'])
            if is_working:
                container['model_id'] = model_id
                working_containers.append(container)
        
        if not working_containers:
            logging.error("No working vLLM endpoints found")
            return None
        
        # Sort by preference
        if prefer_teacher:
            # Prefer teacher models
            teachers = [c for c in working_containers if c['type'] == 'teacher']
            if teachers:
                return teachers[0]
        
        # Return any working model
        return working_containers[0]
    
    def initialize(self, preferred_port: Optional[int] = None) -> bool:
        """Initialize the model manager and select a model"""
        logging.info("Initializing vLLM Model Manager...")
        logging.info(f"USE_REMOTE_MODELS: {self.use_remote_models}")
        logging.info(f"Running in Docker: {self.is_docker}")
        logging.info(f"Using host: {self.get_host()}")
        
        # If a specific port is requested, try it first
        if preferred_port:
            is_working, model_id = self.test_vllm_endpoint(preferred_port)
            if is_working:
                self.selected_port = preferred_port
                self.selected_model = model_id
                host = self.get_host()
                self.client = OpenAI(
                    base_url=f"http://{host}:{preferred_port}/v1",
                    api_key="not-needed"
                )
                logging.info(f"Using requested model on port {preferred_port}")
                return True
        
        # Detect all running containers
        containers = self.detect_running_containers()
        
        if not containers:
            logging.error("No vLLM containers detected.")
            if self.use_remote_models:
                logging.error("Please ensure SSH port forwarding is active:")
                logging.error("ssh -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8003:localhost:8003 user@remote-host")
            else:
                logging.error("Please ensure Docker containers are running locally.")
            return False
        
        # Select the best model
        selected = self.select_best_model(containers, prefer_teacher=True)
        
        if not selected:
            logging.error("No working vLLM models found")
            return False
        
        self.selected_port = selected['port']
        self.selected_model = selected.get('model_id', '/model')
        host = self.get_host()
        self.client = OpenAI(
            base_url=f"http://{host}:{self.selected_port}/v1",
            api_key="not-needed"
        )
        
        logging.info(f"Selected model: {selected['name']} on port {self.selected_port}")
        return True
    
    def generate_teacher_answer(self, question: str) -> str:
        """Generate a teacher answer using the selected model"""
        if not self.client:
            raise RuntimeError("Model not initialized")
            
        # Teacher prompt template
        prompt = f"""You are a helpful assistant providing accurate and detailed answers.

Question: {question}

Please provide a comprehensive answer that is informative and educational."""

        try:
            response = self.client.completions.create(
                model=self.selected_model,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                stop=["\n\n", "Question:", "Q:"]
            )
            
            answer = response.choices[0].text.strip()
            return answer
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Generate teacher-student pairs from input data using vLLM")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--port", type=int, default=None, help="Specific vLLM port to use")
    parser.add_argument("--domain-context", help="Optional domain context file")
    
    args = parser.parse_args()
    
    logging.info("=== Teacher Pair Generation Script (vLLM SSH) Started ===")
    
    # Initialize model manager
    model_manager = VLLMModelManager()
    
    if not model_manager.initialize(preferred_port=args.port):
        logging.error("Failed to initialize vLLM model manager")
        logging.error("Please ensure:")
        logging.error("1. SSH port forwarding is active (if using remote models)")
        logging.error("2. vLLM containers are started")
        logging.error("3. Containers are properly configured with port mappings")
        sys.exit(1)
    
    # Load domain context if provided
    domain_context = ""
    if args.domain_context and os.path.exists(args.domain_context):
        try:
            with open(args.domain_context, 'r') as f:
                domain_context = f.read().strip()
                logging.info(f"Loaded domain context from {args.domain_context}")
        except Exception as e:
            logging.warning(f"Could not load domain context: {e}")
    
    # Process input file
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Load input data
    input_data = []
    try:
        with open(args.input, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    input_data.append(json.loads(line))
        logging.info(f"Loaded {len(input_data)} items from input file")
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Limit samples if specified
    if args.samples and args.samples < len(input_data):
        input_data = input_data[:args.samples]
        logging.info(f"Processing first {args.samples} samples")
    
    # Process each item
    output_data = []
    errors = 0
    
    for idx, item in enumerate(tqdm(input_data, desc="Generating teacher-student pairs")):
        try:
            # Extract question
            question = None
            if 'prompt' in item:
                question = item['prompt']
            elif 'instruction' in item:
                question = item['instruction']
            elif 'question' in item:
                question = item['question']
            elif 'content' in item:
                question = item['content']
            else:
                logging.warning(f"Item {idx} has no recognizable question field")
                continue
            
            # Add domain context if available
            if domain_context:
                question = f"{domain_context}\n\n{question}"
            
            # Generate teacher answer
            teacher_answer = model_manager.generate_teacher_answer(question)
            
            # Create output item
            output_item = {
                "instruction": question,
                "input": "",
                "output": teacher_answer,
                "source": "teacher_generation",
                "metadata": {
                    "original_item": item,
                    "model_port": model_manager.selected_port,
                    "model_id": model_manager.selected_model
                }
            }
            
            output_data.append(output_item)
            
        except Exception as e:
            logging.error(f"Error processing item {idx}: {e}")
            errors += 1
            if errors > len(input_data) * 0.1:  # Stop if more than 10% errors
                logging.error("Too many errors, stopping")
                break
    
    # Save output
    try:
        with open(args.output, 'w') as f:
            for item in output_data:
                f.write(json.dumps(item) + '\n')
        logging.info(f"Successfully saved {len(output_data)} teacher-student pairs to {args.output}")
    except Exception as e:
        logging.error(f"Error saving output: {e}")
        sys.exit(1)
    
    logging.info("=== Teacher Pair Generation Complete ===")
    logging.info(f"Processed: {len(output_data)} pairs")
    logging.info(f"Errors: {errors}")

if __name__ == "__main__":
    main()