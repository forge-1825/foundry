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
    """Manages vLLM model detection and selection"""
    
    def __init__(self):
        self.available_models = []
        self.selected_model = None
        self.selected_port = None
        self.client = None
        
    def detect_running_containers(self) -> List[Dict[str, any]]:
        """Detect all running Docker containers with vLLM models"""
        containers = []
        
        try:
            # Get all running containers
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
                    
                    # Check if this is a vLLM container
                    if 'vllm' in image.lower() or 'vllm' in name.lower():
                        # Extract port mapping
                        port_match = None
                        if '8000->80' in ports or ':8000->' in ports:
                            port_match = 8000
                        elif '8001->80' in ports or ':8001->' in ports:
                            port_match = 8001
                        elif '8002->80' in ports or ':8002->' in ports:
                            port_match = 8002
                        else:
                            # Try to extract any port mapping
                            import re
                            match = re.search(r':(\d+)->', ports)
                            if match:
                                port_match = int(match.group(1))
                        
                        if port_match:
                            containers.append({
                                'name': name,
                                'image': image,
                                'port': port_match,
                                'type': self._determine_model_type(name, image)
                            })
                            
            logging.info(f"Detected {len(containers)} vLLM containers")
            return containers
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Docker command failed: {e}")
            return []
        except Exception as e:
            logging.error(f"Error detecting containers: {e}")
            return []
    
    def _determine_model_type(self, name: str, image: str) -> str:
        """Determine if model is teacher or student based on name/image"""
        combined = (name + " " + image).lower()
        
        if any(teacher in combined for teacher in ['phi4', 'phi-4', 'llama', 'mixtral', 'gpt']):
            return 'teacher'
        elif any(student in combined for student in ['phi3', 'phi-3', 'phi2', 'phi-2']):
            return 'student'
        else:
            return 'unknown'
    
    def test_vllm_endpoint(self, port: int, timeout: int = 30) -> Tuple[bool, Optional[str]]:
        """Test if vLLM endpoint is responding and get model info"""
        logging.info(f"Testing vLLM endpoint on port {port}...")
        
        try:
            client = OpenAI(
                base_url=f"http://localhost:{port}/v1",
                api_key="not-needed"
            )
            
            # Try to get model list first
            try:
                import requests
                response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
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
                model="/model",  # Default model path
                prompt="Hello",
                max_tokens=1,
                temperature=0.1
            )
            
            logging.info(f"vLLM server on port {port} is responding")
            return True, "/model"
            
        except Exception as e:
            logging.warning(f"vLLM server on port {port} not responding: {e}")
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
        
        # If a specific port is requested, try it first
        if preferred_port:
            is_working, model_id = self.test_vllm_endpoint(preferred_port)
            if is_working:
                self.selected_port = preferred_port
                self.selected_model = model_id
                self.client = OpenAI(
                    base_url=f"http://localhost:{preferred_port}/v1",
                    api_key="not-needed"
                )
                logging.info(f"Using requested model on port {preferred_port}")
                return True
        
        # Detect all running containers
        containers = self.detect_running_containers()
        
        if not containers:
            logging.error("No vLLM containers detected. Please ensure Docker containers are running.")
            return False
        
        # Select the best model
        selected = self.select_best_model(containers, prefer_teacher=True)
        
        if not selected:
            logging.error("No working vLLM models found")
            return False
        
        self.selected_port = selected['port']
        self.selected_model = selected.get('model_id', '/model')
        self.client = OpenAI(
            base_url=f"http://localhost:{self.selected_port}/v1",
            api_key="not-needed"
        )
        
        logging.info(f"Selected model: {selected['name']} on port {self.selected_port}")
        return True
    
    def generate_completion(self, prompt: str, max_tokens: int = 250, temperature: float = 0.7) -> str:
        """Generate a completion using the selected model"""
        if not self.client:
            raise ValueError("Model manager not initialized")
        
        try:
            response = self.client.completions.create(
                model=self.selected_model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logging.error(f"Error generating completion: {e}")
            raise

def generate_teacher_output(prompt: str, model_manager: VLLMModelManager, retries: int = 3, delay: int = 5) -> str:
    """
    Query the teacher model via vLLM's OpenAI-compatible API.
    Retries the request up to 'retries' times with a 'delay' between attempts.
    Returns the teacher model's output as a string.
    """
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Sending prompt to vLLM teacher model (attempt {attempt})...")
            output = model_manager.generate_completion(prompt)
            logging.info("Teacher model returned output successfully.")
            return output
        except Exception as e:
            logging.error(f"Error querying vLLM teacher model (attempt {attempt}): {e}")
            if attempt < retries:
                logging.info(f"Retrying request in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("Max retries reached. Returning empty response.")
                return ""

def main():
    parser = argparse.ArgumentParser(description="Generate teacher pairs using vLLM models")
    parser.add_argument("--input-file", type=str, default="Output/enriched_data.json",
                       help="Path to the enriched data JSON file")
    parser.add_argument("--output-file", type=str, default="Output/teacher_pairs.json",
                       help="Path to save the teacher pairs")
    parser.add_argument("--port", type=int, default=None,
                       help="Specific port to use for vLLM server (optional)")
    parser.add_argument("--max-pairs", type=int, default=100,
                       help="Maximum number of pairs to generate")
    parser.add_argument("--domain-context", type=str, default=None,
                       help="Optional domain context file")
    
    args = parser.parse_args()
    
    logging.info("=== Teacher Pair Generation Script (vLLM Improved) Started ===")
    
    # Initialize model manager
    model_manager = VLLMModelManager()
    
    if not model_manager.initialize(preferred_port=args.port):
        logging.error("Failed to initialize vLLM model manager")
        logging.error("Please ensure:")
        logging.error("1. Docker is running")
        logging.error("2. vLLM containers are started")
        logging.error("3. Containers are properly configured with port mappings")
        sys.exit(1)
    
    # Load domain context if provided
    domain_context = ""
    if args.domain_context and os.path.exists(args.domain_context):
        try:
            with open(args.domain_context, 'r', encoding='utf-8') as f:
                domain_context = f.read()
            logging.info(f"Loaded domain context from {args.domain_context}")
        except Exception as e:
            logging.warning(f"Could not load domain context: {e}")
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logging.error(f"Input file not found: {args.input_file}")
        logging.error("Please run the Content Extraction & Enrichment step first")
        sys.exit(1)
    
    # Load enriched data
    logging.info(f"Loading enriched data from {args.input_file}...")
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            enriched_data = json.load(f)
        logging.info(f"Loaded {len(enriched_data)} enriched records")
    except Exception as e:
        logging.error(f"Error loading enriched data: {e}")
        sys.exit(1)
    
    # Generate teacher pairs
    teacher_pairs = []
    errors = []
    
    logging.info(f"Generating teacher pairs (max: {args.max_pairs})...")
    
    with tqdm(total=min(len(enriched_data), args.max_pairs), desc="Generating pairs") as pbar:
        for i, record in enumerate(enriched_data):
            if i >= args.max_pairs:
                break
            
            try:
                # Extract content from the record
                content = record.get('content', '')
                summary = record.get('summary', '')
                entities = record.get('entities', [])
                keywords = record.get('keywords', [])
                
                # Skip empty content
                if not content and not summary:
                    logging.debug(f"Skipping record {i}: No content")
                    continue
                
                # Create a prompt for the teacher model
                prompt = f"""Given the following technical content, generate a comprehensive analysis and explanation.

Content: {content[:1000]}...

Summary: {summary}

Key Entities: {', '.join(entities[:10])}

Keywords: {', '.join(keywords[:10])}

{f"Domain Context: {domain_context[:500]}" if domain_context else ""}

Please provide:
1. A detailed explanation of the main concepts
2. Technical insights and best practices
3. Potential applications or use cases
4. Related topics for further exploration

Analysis:"""

                # Generate teacher output
                teacher_output = generate_teacher_output(prompt, model_manager)
                
                if teacher_output:
                    # Create the teacher pair
                    pair = {
                        'id': f"pair_{i}",
                        'input': prompt,
                        'output': teacher_output,
                        'metadata': {
                            'source_id': record.get('id', f'record_{i}'),
                            'entities': entities,
                            'keywords': keywords,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                    teacher_pairs.append(pair)
                    pbar.update(1)
                else:
                    errors.append(f"Failed to generate output for record {i}")
                    
            except Exception as e:
                logging.error(f"Error processing record {i}: {e}")
                errors.append(f"Error processing record {i}: {str(e)}")
    
    # Save teacher pairs
    if teacher_pairs:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to file
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(teacher_pairs, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully saved {len(teacher_pairs)} teacher pairs to {args.output_file}")
        except Exception as e:
            logging.error(f"Error saving teacher pairs: {e}")
            sys.exit(1)
    else:
        logging.error("No teacher pairs were generated")
        sys.exit(1)
    
    # Report errors if any
    if errors:
        logging.warning(f"Encountered {len(errors)} errors during generation:")
        for error in errors[:10]:  # Show first 10 errors
            logging.warning(f"  - {error}")
    
    # Summary
    logging.info("=== Teacher Pair Generation Complete ===")
    logging.info(f"Total pairs generated: {len(teacher_pairs)}")
    logging.info(f"Total errors: {len(errors)}")
    logging.info(f"Output saved to: {args.output_file}")
    logging.info(f"Model used: {model_manager.selected_model} on port {model_manager.selected_port}")

if __name__ == "__main__":
    main()