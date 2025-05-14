import json
import os
import sys
import time
import logging
import subprocess
from tqdm import tqdm
from openai import OpenAI

# Required Docker containers and their ports
CONTAINER_CONFIG = {
    "phi4_gptq_vllm": 8000  # Primary container for inference
}

REQUIRED_CONTAINERS = list(CONTAINER_CONFIG.keys())

# Configure logging to output messages with timestamps.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Configure OpenAI client to point to the locally running vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Make sure your teacher model container is accessible at this endpoint
    api_key="not-needed"  # vLLM doesn't require an API key
)

def check_vllm_server(max_retries=36, delay=10):
    """
    Check if the vLLM server is responding.
    Retries for up to 6 minutes (36 retries * 10 second delay) by default.
    Returns True if server is responding, False otherwise.
    """
    logging.info("Checking if vLLM server is responding...")
    for attempt in range(1, max_retries + 1):
        try:
            # Try a simple completion request
            response = client.completions.create(
                model="jakiAJK/microsoft-phi-4_GPTQ-int4",
                prompt="test",
                max_tokens=1,
                temperature=0.7,
                n=1,
                stop=None
            )
            logging.info("vLLM server is responding!")
            return True
        except Exception as e:
            if attempt < max_retries:
                logging.warning(f"vLLM server not ready (attempt {attempt}/{max_retries}). Waiting {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error("vLLM server failed to respond after maximum retries")
                return False
    return False

def verify_container_ports():
    """
    Verify that containers are running on their expected ports.
    Returns True if ports are correct, False otherwise.
    """
    try:
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}\t{{.Ports}}'], 
                              capture_output=True, text=True, check=True)
        container_ports = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    name, ports = parts
                    # Extract port from format like "0.0.0.0:8000->8000/tcp"
                    if '->' in ports:
                        port = ports.split(':')[1].split('->')[0]
                        container_ports[name] = int(port)
        
        for container, expected_port in CONTAINER_CONFIG.items():
            if container in container_ports:
                actual_port = container_ports[container]
                if actual_port != expected_port:
                    logging.error(f"Container {container} is running on port {actual_port}, expected {expected_port}")
                    return False
                logging.info(f"Verified {container} is running on correct port {expected_port}")
        return True
    except Exception as e:
        logging.error(f"Error verifying container ports: {e}")
        return False

def check_container_logs(container, timeout=60):
    """
    Check container logs for server readiness.
    Returns True if server appears ready, False otherwise.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            log_result = subprocess.run(['docker', 'logs', container], 
                                      capture_output=True, text=False, check=True)
            log_text = log_result.stdout.decode('utf-8', errors='ignore')
            if "Uvicorn running on" in log_text:
                logging.info(f"Container {container} vLLM server initialized")
                return True
            time.sleep(5)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error checking logs for container {container}: {e.stderr}")
            return False
    logging.error(f"Container {container} vLLM server not initialized after {timeout} seconds")
    return False

def check_and_start_containers():
    """
    Check if required Docker containers are running and start them if needed.
    Returns True if all containers are running, False if there was an error.
    """
    try:
        logging.info("Checking Docker container status...")
        
        # First check if Docker is running
        try:
            version_check = subprocess.run(['docker', 'version'], 
                                         capture_output=True, text=True, check=True)
            logging.info("Docker is running")
        except subprocess.CalledProcessError:
            logging.error("Docker is not running. Please start Docker first.")
            return False
        
        # Get list of running containers
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                              capture_output=True, text=True, check=True)
        running_containers = result.stdout.strip().split('\n')
        running_containers = [c for c in running_containers if c]  # Remove empty strings
        
        logging.info(f"Currently running containers: {', '.join(running_containers) if running_containers else 'none'}")
        
        # Check container status
        containers_to_start = []
        for container in REQUIRED_CONTAINERS:
            if container not in running_containers:
                containers_to_start.append(container)
                logging.warning(f"Required container '{container}' is not running")
            else:
                logging.info(f"Required container '{container}' is already running")
        
        if containers_to_start:
            logging.info(f"Attempting to start {len(containers_to_start)} containers...")
            
            # Start any containers that aren't running
            for container in containers_to_start:
                logging.info(f"Starting container: {container}")
                try:
                    result = subprocess.run(['docker', 'start', container], 
                                          capture_output=True, text=True, check=True)
                    logging.info(f"Successfully started container: {container}")
                    
                    # Give container time to initialize
                    init_wait = 60 if "vllm" in container.lower() else 5
                    logging.info(f"Waiting {init_wait} seconds for {container} to initialize...")
                    time.sleep(init_wait)
                    
                    # Check container logs for readiness
                    if "vllm" in container.lower():
                        if not check_container_logs(container):
                            return False
                    
                    # Verify container is now running
                    check_result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}', '--filter', f'name={container}'],
                                                capture_output=True, text=True, check=True)
                    if container in check_result.stdout:
                        logging.info(f"Verified {container} is now running")
                    else:
                        logging.error(f"Container {container} failed to start properly")
                        return False
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to start container {container}: {e.stderr}")
                    return False
        
        if containers_to_start:
            logging.info("All required containers have been started")
        else:
            logging.info("All required containers are already running")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Docker command failed: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error checking/starting Docker containers: {e}")
        return False

def generate_teacher_output(prompt, retries=3, delay=5):
    """
    Query the teacher model via vLLM's OpenAI-compatible API.
    Retries the request up to 'retries' times with a 'delay' between attempts.
    Returns the teacher model's output as a string.
    """
    for attempt in range(1, retries + 1):
        try:
            logging.info("Sending prompt to vLLM teacher model (attempt %d)...", attempt)
            response = client.completions.create(
                model="jakiAJK/microsoft-phi-4_GPTQ-int4",
                prompt=prompt,
                max_tokens=250,
                temperature=0.7,
                n=1,
                stop=None
            )
            teacher_output = response.choices[0].text.strip()
            logging.info("Teacher model returned output successfully.")
            return teacher_output
        except Exception as e:
            logging.error("Error querying vLLM teacher model (attempt %d): %s", attempt, e)
            if attempt < retries:
                logging.info("Retrying request in %.2f seconds...", delay)
                time.sleep(delay)
            else:
                return ""

def main():
    logging.info("=== Teacher Pair Generation Script (vLLM) Started ===")
    
    logging.info("=== Checking Docker Container Status ===")
    logging.info(f"Required containers: {', '.join(REQUIRED_CONTAINERS)}")
    # Check and start required Docker containers
    if not check_and_start_containers():
        logging.error("Failed to ensure all required Docker containers are running.")
        logging.error("Please make sure Docker is running and the containers exist.")
        sys.exit(1)
    
    # Verify container ports
    logging.info("=== Verifying Container Ports ===")
    if not verify_container_ports():
        logging.error("Container port verification failed.")
        logging.error("Please ensure containers are running on their correct ports.")
        sys.exit(1)
    logging.info("=== Container Port Verification Complete ===")
    
    # Wait for vLLM server to be ready
    logging.info("=== Checking vLLM Server Status ===")
    if not check_vllm_server():
        logging.error("vLLM server is not responding.")
        logging.error("Please check the container logs for any issues.")
        sys.exit(1)
    logging.info("=== vLLM Server Check Complete ===")
    
    # Optional: Load domain context from file if available.
    domain_context = ""
    domain_context_path = r"G:\NeedInput\domain_context.txt"
    if os.path.exists(domain_context_path):
        try:
            with open(domain_context_path, "r", encoding="utf-8") as f:
                domain_context = f.read()
            logging.info(f"Loaded domain context from {domain_context_path} ({len(domain_context)} characters)")
        except Exception as e:
            logging.error(f"Error loading domain context: {e}")
    
    # Paths for enriched data
    input_path = r"G:\NeedInput\enriched_data.json"
    output_path = r"G:\NeedInput\teacher_pairs_data.json"
    
    logging.info("Using input file: %s", input_path)
    if not os.path.exists(input_path):
        logging.error("Input file not found: %s. Exiting.", input_path)
        sys.exit(1)
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logging.info("Loaded %d records from %s.", len(dataset), input_path)
    except Exception as e:
        logging.error("Error loading JSON file: %s", e)
        sys.exit(1)
    
    if not dataset:
        logging.error("Dataset is empty. Exiting.")
        sys.exit(1)
    
    # Enhanced prompt template with hierarchical context encoding and chain-of-thought reasoning.
    prompt_template = (
        "Using the following domain context and document excerpt, perform hierarchical context encoding and chain-of-thought reasoning to extract detailed technical requirements.\n\n"
        "Domain Context:\n{domain_context}\n\n"
        "Document Excerpt:\n{input_text}\n\n"
        "Chain-of-Thought: Describe step-by-step your reasoning process to identify key technical requirements from the above text.\n\n"
        "Detailed Requirements:"
    )
    
    teacher_pairs = []
    max_prompt_length = 1000  # Limit prompt length to avoid overwhelming the teacher model
    
    logging.info("Beginning teacher output generation for each record...")
    
    try:
        for idx, record in enumerate(tqdm(dataset, desc="Processing records"), start=1):
            input_text = record.get("summary", "") or record.get("cleaned_text", "") or record.get("raw_content", "")
            path = record.get("path", "N/A")
            if not input_text:
                logging.warning("Record %d (Path: %s) has no input text. Skipping.", idx, path)
                continue

            # Truncate input_text if it's too long
            if len(input_text) > max_prompt_length:
                logging.info("Truncating input text for record %d (Path: %s)", idx, path)
                input_text = input_text[:max_prompt_length]
            
            # Fill the prompt template; include domain context if available.
            prompt = prompt_template.format(domain_context=domain_context, input_text=input_text)
            logging.info("Record %d: Generated prompt for Path %s", idx, path)
            
            teacher_output = generate_teacher_output(prompt)
            if not teacher_output:
                logging.warning("Record %d (Path: %s) returned no teacher output.", idx, path)
            else:
                logging.info("Record %d (Path: %s): Teacher output generated.", idx, path)
            
            teacher_pairs.append({
                "path": path,
                "input": input_text,
                "target": teacher_output
            })
            time.sleep(0.5)  # Optional delay to avoid overloading the server
    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user. Saving progress...")
        # Continue to save what we have so far
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception as e:
        logging.error("Error creating output directory: %s", e)
        sys.exit(1)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(teacher_pairs, f, ensure_ascii=False, indent=4)
        logging.info("Teacher pairs successfully saved to %s.", output_path)
    except Exception as e:
        logging.error("Error saving teacher pairs: %s", e)
    
    logging.info("=== Teacher Pair Generation Script (vLLM) Completed ===")

if __name__ == "__main__":
    main()
