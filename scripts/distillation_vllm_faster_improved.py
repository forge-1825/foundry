import os
import json
import logging
import torch
import torch.nn.functional as F
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Union
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    get_scheduler
)
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from openai import OpenAI
import time

# --------------------------------------------------------------------------
# Logging configuration: log both to console and to a file (overwriting old logs)
# --------------------------------------------------------------------------
# Configure file handler with UTF-8 encoding
file_handler = logging.FileHandler("training_phi2_improved.log", mode='w', encoding='utf-8')

# Configure console handler with error handling for non-UTF-8 characters
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Replace problematic characters with '?' for console output
            msg = self.format(record)
            safe_msg = msg.encode('ascii', 'replace').decode('ascii')
            self.stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Set up logging with both handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        SafeStreamHandler(),
        file_handler
    ]
)

# --------------------------------------------------------------------------
# Check CUDA availability and log GPU details
# --------------------------------------------------------------------------
if torch.cuda.is_available():
    logging.info(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
        logging.info(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.1f}MB")
        logging.info(f"Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.1f}MB")
        logging.info(f"Total memory: {props.total_memory / 1024**2:.1f}MB")
else:
    logging.warning("CUDA is not available. Training will be slow on CPU.")

# --------------------------------------------------------------------------
# Configure OpenAI clients to point to the locally running vLLM servers
# --------------------------------------------------------------------------
# Teacher model (phi4_gptq_vllm) on port 8000
teacher_client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't require an API key
)

# Student model (whiterabbitneo_vllm_new) on port 8001
student_client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"  # vLLM doesn't require an API key
)

# --------------------------------------------------------------------------
# Check if vLLM servers are responding
# --------------------------------------------------------------------------
def check_vllm_server(client, name, max_retries=3, delay=5):
    """Check if a vLLM server is responding."""
    logging.info(f"Checking if {name} vLLM server is responding...")
    for attempt in range(1, max_retries + 1):
        try:
            # Try a simple completion request
            if name == "Teacher (port 8000)":
                response = client.completions.create(
                    model="jakiAJK/microsoft-phi-4_GPTQ-int4",  # Model name from Docker logs
                    prompt="test",
                    max_tokens=1,
                    temperature=0.7,
                    n=1,
                    stop=None
                )
            else:  # Student (port 8001)
                response = client.completions.create(
                    model="TheBloke/WhiteRabbitNeo-13B-AWQ",  # Model name from Docker logs
                    prompt="test",
                    max_tokens=1,
                    temperature=0.7,
                    n=1,
                    stop=None
                )
            logging.info(f"{name} vLLM server is responding!")
            return True
        except Exception as e:
            if attempt < max_retries:
                logging.warning(f"{name} vLLM server not ready (attempt {attempt}/{max_retries}). Waiting {delay} seconds...")
                import time
                time.sleep(delay)
            else:
                logging.error(f"{name} vLLM server failed to respond after maximum retries: {e}")
                return False
    return False

# Check both servers
teacher_server_ok = check_vllm_server(teacher_client, "Teacher (port 8000)")
student_server_ok = check_vllm_server(student_client, "Student (port 8001)")

if not teacher_server_ok or not student_server_ok:
    logging.error("One or both vLLM servers are not responding. Please check Docker containers.")
    exit(1)

# --------------------------------------------------------------------------
# Load teacher pairs (the distilled teacher outputs)
# --------------------------------------------------------------------------
# Try different possible locations for the teacher pairs file
teacher_pairs_paths = [
    r"G:\NeedInput\backup files\Output\teacher_pairs.json",
    r"G:\NeedInput\teacher_pairs_data.json",
    r"G:\NeedInput\teacher_pairs.json",
    r"G:\NeedInput\Output\teacher_pairs.json"
]

# Try to find the teacher pairs file
teacher_pairs_file_found = False
for path in teacher_pairs_paths:
    if os.path.exists(path):
        input_teacher_pairs = path
        logging.info(f"Found teacher pairs file at {path}")
        teacher_pairs_file_found = True
        break

if not teacher_pairs_file_found:
    logging.error("Could not find teacher pairs file. Please check the file paths.")
    logging.error("Looked in the following locations:")
    for path in teacher_pairs_paths:
        logging.error(f"  - {path}")
    logging.error("Creating a small sample dataset for testing...")
    
    # Create a small sample dataset for testing
    teacher_pairs = [
        {"input": "What is machine learning?", "target": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."},
        {"input": "Explain quantum computing.", "target": "Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data."},
        {"input": "What is the capital of France?", "target": "The capital of France is Paris."}
    ]
else:
    # Load the teacher pairs from the found file
    logging.info(f"Loading teacher pairs from {input_teacher_pairs}...")
    try:
        with open(input_teacher_pairs, "r", encoding="utf-8") as f:
            teacher_pairs = json.load(f)
        logging.info(f"Loaded {len(teacher_pairs)} teacher pairs.")
    except Exception as e:
        logging.error(f"Error loading teacher pairs file: {e}")
        logging.error("Creating a small sample dataset for testing...")
        
        # Create a small sample dataset for testing
        teacher_pairs = [
            {"input": "What is machine learning?", "target": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."},
            {"input": "Explain quantum computing.", "target": "Quantum computing is a type of computing that uses quantum-mechanical phenomena, such as superposition and entanglement, to perform operations on data."},
            {"input": "What is the capital of France?", "target": "The capital of France is Paris."}
        ]

# --------------------------------------------------------------------------
# Generate targets using both Phi-4 and WhiteRabbitNeo (Ensemble approach)
# --------------------------------------------------------------------------
def generate_targets_with_ensemble(teacher_pairs, max_entries=30):  # Increased from 20 to 30
    """Generate targets using both Phi-4 and WhiteRabbitNeo, then combine them."""
    logging.info("Generating targets using ensemble approach (Phi-4 + WhiteRabbitNeo)...")
    
    # Count entries with empty targets
    empty_target_count = sum(1 for pair in teacher_pairs if not pair.get("target") or not pair.get("target").strip())
    logging.info(f"Found {empty_target_count} entries with empty targets.")
    
    # Limit the number of entries to process to avoid overloading the server
    entries_to_process = min(empty_target_count, max_entries)
    logging.info(f"Will generate targets for {entries_to_process} entries.")
    
    processed_count = 0
    for i, pair in enumerate(teacher_pairs):
        if not pair.get("target") or not pair.get("target").strip():
            if processed_count >= entries_to_process:
                break
                
            if pair.get("input") and pair.get("input").strip():
                input_text = pair["input"].strip()
                # Add the prompt template to make it clearer what we want
                prompt = f"Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\n{input_text}"
                logging.info(f"Generating ensemble target for entry {i+1}/{len(teacher_pairs)} with input: {input_text[:50]}...")
                
                try:
                    # Generate a completion using the Phi-4 vLLM server
                    phi4_response = teacher_client.completions.create(
                        model="jakiAJK/microsoft-phi-4_GPTQ-int4",
                        prompt=prompt,  # Use the enhanced prompt
                        max_tokens=300,  # Increased from 200 to 300
                        temperature=0.3,  # Reduced from 0.7 to 0.3 for more focused outputs
                        n=1,
                        stop=None
                    )
                    
                    # Extract the generated text from Phi-4
                    phi4_text = phi4_response.choices[0].text.strip()
                    logging.info(f"Generated Phi-4 response: {phi4_text[:50]}...")
                    
                    # Generate a completion using the WhiteRabbitNeo vLLM server
                    wrn_response = student_client.completions.create(
                        model="TheBloke/WhiteRabbitNeo-13B-AWQ",
                        prompt=prompt,  # Use the enhanced prompt
                        max_tokens=300,  # Increased from 200 to 300
                        temperature=0.3,  # Reduced from 0.7 to 0.3 for more focused outputs
                        n=1,
                        stop=None
                    )
                    
                    # Extract the generated text from WhiteRabbitNeo
                    wrn_text = wrn_response.choices[0].text.strip()
                    logging.info(f"Generated WhiteRabbitNeo response: {wrn_text[:50]}...")
                    
                    # Store both responses separately
                    pair["phi4_target"] = phi4_text
                    pair["wrn_target"] = wrn_text
                    
                    # Combine responses (simple approach: use Phi-4 as primary, WhiteRabbitNeo as alternative)
                    # A more sophisticated approach would merge the content
                    pair["target"] = phi4_text
                    
                    # Add a small delay to avoid overloading the servers
                    time.sleep(0.5)
                    
                    processed_count += 1
                except Exception as e:
                    logging.error(f"Error generating ensemble target for entry {i+1}: {e}")
                    # Fallback to just Phi-4 if ensemble fails
                    try:
                        # Generate a completion using just the Phi-4 vLLM server
                        response = teacher_client.completions.create(
                            model="jakiAJK/microsoft-phi-4_GPTQ-int4",
                            prompt=prompt,  # Use the enhanced prompt
                            max_tokens=300,  # Increased from 200 to 300
                            temperature=0.3,  # Reduced from 0.7 to 0.3 for more focused outputs
                            n=1,
                            stop=None
                        )
                        
                        # Extract the generated text
                        generated_text = response.choices[0].text.strip()
                        pair["target"] = generated_text
                        logging.info(f"Fallback to Phi-4 only: {generated_text[:50]}...")
                        
                        processed_count += 1
                    except Exception as e2:
                        logging.error(f"Error in fallback generation for entry {i+1}: {e2}")
            else:
                # If input is also empty, use a default target
                pair["target"] = "This is a placeholder target for training purposes."
                processed_count += 1
    
    logging.info(f"Generated ensemble targets for {processed_count} entries.")
    return teacher_pairs

# --------------------------------------------------------------------------
# Augment data with additional examples from WhiteRabbitNeo
# --------------------------------------------------------------------------
def augment_data_with_whiterabbitneo(teacher_pairs, num_new_examples=20):  # Increased from 10 to 20
    """Generate additional training examples using WhiteRabbitNeo."""
    logging.info(f"Augmenting data with {num_new_examples} new examples from WhiteRabbitNeo...")
    
    augmented_pairs = teacher_pairs.copy()
    
    # Create new prompts focused on technical requirements extraction
    # These are aligned with the domain focus of teacher_pair_generation_vllm.py
    # Added more explicit instructions to format as a numbered list
    new_prompts = [
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Cisco Catalyst 9300 Series switches are enterprise-class access switches that provide full convergence between wired and wireless networks. The switches offer 24 or 48 ports of Gigabit Ethernet and Multigigabit Ethernet, with optional 10 Gigabit or 40 Gigabit uplinks. They support up to 384 ports of Gigabit Ethernet with StackWise technology.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Dell PowerEdge R740 is a 2-socket, 2U rack server designed for complex workloads using highly scalable memory and network options. It features up to 28 cores per processor, up to 16 DIMMs, and choice of NVMe drives. It supports up to 3 double-width GPUs or 6 single-width GPUs.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Juniper Networks MX204 Universal Routing Platform is a cloud-grade routing platform that offers 400 Gbps of throughput. It features high-density 100GbE, supporting up to 4 ports, as well as high-density 10GbE, supporting up to 24 ports. The MX204 is designed for space-constrained service provider and enterprise deployments.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe HPE ProLiant DL380 Gen10 server is a secure 2P 2U server that delivers performance and expandability. It supports Intel Xeon Scalable processors with up to a 60% performance gain. It features up to 3.0 TB of 2666 MT/s HPE DDR4 SmartMemory and supports 12 Gb/s SAS and up to 20 NVMe drive options.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Arista 7280R Series are fixed configuration 10/25/40/50/100GbE systems built for the highest performance environments. They combine scalable L2 and L3 resources with advanced features for network monitoring, precision timing, and network virtualization. They offer up to 60 QSFP100 ports or 48 SFP25 and 8 QSFP100 ports.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Palo Alto Networks PA-5250 is a next-generation firewall appliance that offers 20 Gbps of throughput with security services. It features 8 Gbps of threat prevention throughput and 10 Gbps of IPsec VPN throughput. The system supports up to 4 million sessions and 214,000 new connections per second.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Fortinet FortiGate 600E is a next-generation firewall that delivers high performance with a security processor. It offers 36 Gbps firewall throughput, 5.5 Gbps IPsec VPN throughput, and 7 Gbps threat protection throughput. It features multiple high-speed interfaces including 16 GE RJ45 ports, 16 GE SFP slots, and 2 10GE SFP+ slots.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Ubiquiti UniFi Switch Pro 48 PoE is a managed Gigabit switch with 48 auto-sensing PoE+ ports and 4 10G SFP+ ports. It delivers up to 600W PoE with a total switching capacity of 140 Gbps. The switch features Layer 3 capabilities including static routing, and supports 802.1X authentication and VLANs.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Ruckus ICX 7150 is a stackable enterprise switch that delivers enterprise-class features. It offers 24 or 48 ports of 10/100/1000 Mbps and up to 8 ports of 10 GbE for uplinks or stacking. The switch provides up to 740 watts of PoE power and supports up to 12 switches in a stack with up to 576 ports.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Extreme Networks X465 is a versatile 1/10/25/40/100 GbE switch designed for enterprise edge and aggregation deployments. It offers up to 48 ports of 1/10 GbE, up to 12 ports of 25 GbE, and up to 4 ports of 40/100 GbE. The switch supports VSF stacking of up to 8 units and provides up to 1.76 Tbps of switching capacity.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Brocade G620 is a high-density storage networking switch designed for large enterprise environments. It provides up to 64 ports of 32 Gbps Fibre Channel in a 1U form factor. The switch delivers 2 Tbps of aggregate bandwidth and features auto-sensing 4, 8, 16, and 32 Gbps capabilities with support for NVMe over Fibre Channel.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Mellanox SN2700 is a high-performance Ethernet switch with 32 ports of 100GbE in a compact 1U form factor. It delivers up to 6.4 Tbps of switching capacity with a forwarding rate of 4.76 Bpps. The switch features a shared buffer of 16MB and latency as low as 300ns.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Aruba 8320 Switch Series is a family of high-performance 10/25/40/100 GbE switches designed for enterprise core and aggregation deployments. It offers 48 ports of 10/25 GbE and 6 ports of 40/100 GbE in a 1U form factor. The switch provides 2.0 Tbps of switching capacity and up to 1,500 Mpps of throughput.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Cisco Nexus 9336C-FX2 is a 1RU switch that supports 7.2 Tbps of bandwidth and 4 billion packets per second. It offers 36 40/100-Gbps QSFP28 ports. All ports support wire-rate performance. The switch is designed for high-performance applications in NX-OS mode or as a leaf or spine in ACI mode.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Juniper Networks EX4650 is a 1U 25GbE/100GbE data center switch that offers 48 25-Gbps SFP28 ports and 8 100-Gbps QSFP28 ports. It delivers 4 Tbps of throughput and 2.97 Bpps of forwarding capacity. The switch supports EVPN-VXLAN and features a deep buffer of 32 GB for lossless operation.",
        
        # Adding more examples with security-focused requirements
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Cisco ASA 5500-X Series is a next-generation firewall that provides comprehensive security with integrated IPS capabilities. It offers up to 4 Gbps of firewall throughput, 1 Gbps of IPS throughput, and supports up to 500,000 concurrent sessions. The appliance features 8 Gigabit Ethernet interfaces and optional fiber interfaces for high-speed connectivity.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Juniper SRX340 Services Gateway is a next-generation firewall that provides security and networking services for branch offices. It delivers up to 5 Gbps of firewall throughput, 800 Mbps of IPsec VPN throughput, and 500 Mbps of IPS throughput. The gateway features 16 GE ports, 4 SFP ports, and supports up to 64,000 concurrent sessions.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Check Point 5800 Appliance is a security gateway that provides enterprise-grade protection against Gen V cyber attacks. It offers up to 20 Gbps of firewall throughput, 3.4 Gbps of threat prevention throughput, and 8 Gbps of IPsec VPN throughput. The appliance features 8 copper GE ports, 8 SFP GE ports, and 2 SFP+ 10GE ports.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Fortinet FortiAuthenticator-400E is an identity and access management appliance that provides centralized authentication services. It supports up to 5,000 concurrent users, 1,000 concurrent SSL VPN users, and 2,000 concurrent RADIUS clients. The appliance features 4 GE RJ45 ports, 2 USB ports, and 1 TB of storage.",
        
        "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\nThe Cisco Catalyst 9800-40 Wireless Controller is a fixed wireless controller that supports up to 2,000 access points and 32,000 clients. It offers 40 Gbps of throughput and features 8 10G SFP+ ports. The controller supports Wi-Fi 6 (802.11ax) and provides advanced security features including Encrypted Traffic Analytics and Cisco Software-Defined Access."
    ]
    
    added_count = 0
    for prompt in new_prompts[:num_new_examples]:
        try:
            logging.info(f"Generating augmentation example with prompt: {prompt[:50]}...")
            
            # Get WhiteRabbitNeo's response
            wrn_response = student_client.completions.create(
                model="TheBloke/WhiteRabbitNeo-13B-AWQ",
                prompt=prompt,
                max_tokens=300,  # Increased from 200 to 300
                temperature=0.3,  # Reduced from 0.7 to 0.3 for more focused outputs
                n=1,
                stop=None
            ).choices[0].text.strip()
            
            # Also get Phi-4's response for comparison
            phi4_response = teacher_client.completions.create(
                model="jakiAJK/microsoft-phi-4_GPTQ-int4",
                prompt=prompt,
                max_tokens=300,  # Increased from 200 to 300
                temperature=0.3,  # Reduced from 0.7 to 0.3 for more focused outputs
                n=1,
                stop=None
            ).choices[0].text.strip()
            
            # Add to training data
            augmented_pairs.append({
                "input": prompt,
                "target": phi4_response,  # Use Phi-4 as the primary target
                "wrn_target": wrn_response,
                "phi4_target": phi4_response,
                "source": "augmentation"
            })
            
            added_count += 1
            logging.info(f"Added augmentation example {added_count}/{num_new_examples}")
            
            # Add a small delay to avoid overloading the servers
            time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"Error augmenting data with WhiteRabbitNeo: {e}")
    
    logging.info(f"Added {added_count} new examples to the training data.")
    return augmented_pairs

# Generate targets using ensemble approach
teacher_pairs = generate_targets_with_ensemble(teacher_pairs)

# Augment data with additional examples
teacher_pairs = augment_data_with_whiterabbitneo(teacher_pairs)

# --------------------------------------------------------------------------
# Use input as target for remaining empty entries
# --------------------------------------------------------------------------
for pair in teacher_pairs:
    if not pair.get("target") or not pair.get("target").strip():
        if pair.get("input") and pair.get("input").strip():
            pair["target"] = pair["input"].strip()
        else:
            pair["target"] = "This is a placeholder target for training purposes."

# --------------------------------------------------------------------------
# Load Phi-2 as the student model with 4-bit quantization
# --------------------------------------------------------------------------
logging.info("Loading student model for fine-tuning...")

# Create an offload folder for any weights that must be stored on disk
offload_folder = "./model_offload"
os.makedirs(offload_folder, exist_ok=True)

# Configure 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Check if CUDA is available and print PyTorch configuration
logging.info(f"PyTorch version: {torch.__version__}")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    logging.warning("CUDA is not available. Please check your PyTorch installation and CUDA drivers.")
    logging.warning("You can install PyTorch with CUDA support using:")
    logging.warning("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    logging.warning("Continuing with CPU, but training will be extremely slow...")

# Since we can't load the AWQ model locally without autoawq, we'll use Phi-2 for training
# while still using WhiteRabbitNeo for inference through the vLLM API
logging.info("Using Phi-2 model for training since we can't load WhiteRabbitNeo-AWQ without autoawq")
logging.info("WhiteRabbitNeo will still be used for inference through the vLLM API")

model_name = "microsoft/phi-2"
try:
    # Load tokenizer
    logging.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization if CUDA is available, otherwise use FP32
    logging.info(f"Loading model from {model_name}")
    if torch.cuda.is_available():
        student_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            offload_folder=offload_folder,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    else:
        student_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder=offload_folder,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    
    logging.info(f"Loaded student model {model_name} successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    logging.error("Please check your model path and PyTorch installation.")
    exit(1)

student_model.gradient_checkpointing_enable()
student_model.config.use_cache = False

print("Student model loaded successfully on", next(student_model.parameters()).device)
if torch.cuda.is_available():
    logging.info(f"Model device: {next(student_model.parameters()).device}")
    logging.info(f"Initial GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
logging.info(f"Student model loaded successfully.")

# --------------------------------------------------------------------------
# Split teacher pairs into train and validation sets for early stopping
# --------------------------------------------------------------------------
val_size = min(20, int(0.05 * len(teacher_pairs)))  # Reduced from 30 to 20 to have more training data
train_teacher_pairs = teacher_pairs[:-val_size]
val_teacher_pairs = teacher_pairs[-val_size:]
logging.info(f"Training set size: {len(train_teacher_pairs)}")
logging.info(f"Validation set size: {len(val_teacher_pairs)}")

# --------------------------------------------------------------------------
# Define data collator and custom dataset for distillation training.
# --------------------------------------------------------------------------
@dataclass
class DataCollatorForDistillation:
    tokenizer: AutoTokenizer
    max_length: int = 256  # Increased from 128 to 256 for longer sequences
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, teacher_pairs, tokenizer, max_length=256):  # Increased from 128 to 256
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logging.info("Tokenizer settings:")
        logging.info(f"BOS token: {tokenizer.bos_token} ({tokenizer.bos_token_id})")
        logging.info(f"EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
        logging.info(f"PAD token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
        
        for pair in teacher_pairs:
            if pair.get("target") and len(pair["target"].strip()) > 0:
                try:
                    target_text = pair["target"].strip()
                    if target_text:
                        if len(self.data) == 0:
                            # Safely log the first example with ASCII-only characters
                            safe_text = target_text[:100].encode('ascii', 'replace').decode('ascii')
                            logging.info(f"Processing first example: {safe_text}...")
                        
                        encoded = self.tokenizer(
                            target_text,
                            truncation=True,
                            max_length=self.max_length - 2,  # Reserve space for special tokens
                            padding=False,
                            return_tensors=None
                        )
                        
                        # Manually add special tokens
                        input_ids = [self.tokenizer.bos_token_id] + encoded["input_ids"] + [self.tokenizer.eos_token_id]
                        attention_mask = [1] + encoded["attention_mask"] + [1]
                        padding_length = self.max_length - len(input_ids)
                        if padding_length > 0:
                            input_ids += [self.tokenizer.pad_token_id] * padding_length
                            attention_mask += [0] * padding_length
                        
                        # Create tensors on CPU; Accelerator will move them later.
                        input_ids = torch.tensor(input_ids, device='cpu')
                        attention_mask = torch.tensor(attention_mask, device='cpu')
                        
                        if len(self.data) == 0:
                            logging.info(f"First example tokens: {input_ids[:10].tolist()}...")
                            # Safely log the decoded text with ASCII-only characters
                            decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)[:100]
                            safe_decoded = decoded_text.encode('ascii', 'replace').decode('ascii')
                            logging.info(f"First example decoded: {safe_decoded}...")
                        
                        self.data.append({
                            "input_ids": input_ids,
                            "attention_mask": attention_mask
                        })
                except Exception as e:
                    logging.error(f"Error processing example: {e}")
                    continue
        
        logging.info(f"Dataset created with {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = DistillationDataset(train_teacher_pairs, tokenizer)
val_dataset = DistillationDataset(val_teacher_pairs, tokenizer)

os.makedirs("./distilled_model_phi2_improved", exist_ok=True)

# --------------------------------------------------------------------------
# Apply PEFT for memory-efficient training: prepare model for k-bit training and apply LoRA.
# --------------------------------------------------------------------------
logging.info("Preparing model for k-bit training...")
student_model = prepare_model_for_kbit_training(student_model)

# Use higher rank for better learning capacity
lora_config = LoraConfig(
    r=32,  # Increased from 16 to 32 for better learning capacity
    lora_alpha=64,  # Increased from 32 to 64 to match the higher rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],  # Added fc1 and fc2 for MLP layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

logging.info("Applying LoRA configuration...")
student_model = get_peft_model(student_model, lora_config)
student_model.print_trainable_parameters()

# --------------------------------------------------------------------------
# Initialize Accelerator with GPU optimizations and mixed precision.
# --------------------------------------------------------------------------
# Configure accelerator based on available parameters
try:
    # Try with newer parameters first
    accelerator = Accelerator(
        gradient_accumulation_steps=16,  # Reduced from 24 to 16 for more frequent updates
        mixed_precision="bf16" if torch.cuda.is_available() else None
    )
    logging.info("Using accelerator with mixed precision")
except Exception as e:
    # Last resort - minimal configuration
    logging.warning(f"Error configuring accelerator with mixed precision: {e}")
    accelerator = Accelerator(
        gradient_accumulation_steps=16  # Reduced from 24 to 16 for more frequent updates
    )
    logging.info("Using accelerator with minimal configuration")

# Create data loaders with slightly increased batch size
train_dataloader = DataLoader(
    train_dataset,
    batch_size=2,  # Increased from 1 to 2 for slightly faster training
    shuffle=True,
    collate_fn=DataCollatorForDistillation(tokenizer=tokenizer)
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=2,  # Increased from 1 to 2 for consistency
    shuffle=False,
    collate_fn=DataCollatorForDistillation(tokenizer=tokenizer)
)

# Calculate total training steps with increased epochs
num_epochs = 15  # Increased from 8 to 15 for more training
num_training_steps = num_epochs * len(train_dataloader)

# Create optimizer and learning rate scheduler
optimizer = AdamW(
    student_model.parameters(), 
    lr=1e-4,  # Reduced from 2e-4 to 1e-4 for more stable training
    eps=1e-4,
    weight_decay=0.01
)

lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=50,  # Increased from 25 to 50 for better initialization
    num_training_steps=num_training_steps,
)

# Prepare the model, optimizer, dataloaders, and scheduler with Accelerator.
student_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    student_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

student_model.train()
completed_steps = 0

# --------------------------------------------------------------------------
# Evaluation functions for validation
# --------------------------------------------------------------------------
def evaluate(model, dataloader):
    """Compute average loss on validation data."""
    model.eval()
    losses = []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch["input_ids"].shape[0])))
    
    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses).item()
    except:
        eval_loss = float("inf")
    
    model.train()
    return eval_loss

def evaluate_with_whiterabbitneo(tokenizer, model, val_pairs, max_samples=5):
    """Compare Phi-2 outputs with WhiteRabbitNeo outputs for evaluation."""
    logging.info(f"Evaluating with WhiteRabbitNeo comparison (max {max_samples} samples)...")
    
    # Sample some prompts for comparison
    sample_prompts = []
    for pair in val_pairs[:max_samples*2]:  # Get more than needed in case some fail
        if pair.get("input") and len(sample_prompts) < max_samples:
            # Add the prompt template to make it clearer what we want
            input_text = pair["input"].strip()
            prompt = f"Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\n{input_text}"
            sample_prompts.append(prompt)
    
    if not sample_prompts:
        logging.warning("No valid prompts found for WhiteRabbitNeo evaluation")
        return 0.0
    
    whiterabbitneo_scores = []
    for i, prompt in enumerate(sample_prompts[:max_samples]):
        try:
            # Get Phi-2 (student model) response
            inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=200,  # Use max_new_tokens instead of max_length
                    temperature=0.3,  # Reduced from 0.7 to 0.3 for more focused outputs
                    do_sample=True
                )
            phi2_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text if it's included
            if phi2_response.startswith(prompt):
                phi2_response = phi2_response[len(prompt):].strip()
            
            # Get WhiteRabbitNeo response
            wrn_response = student_client.completions.create(
                model="TheBloke/WhiteRabbitNeo-13B-AWQ",
                prompt=prompt,
                max_tokens=200,
                temperature=0.3,  # Reduced from 0.7 to 0.3 for more focused outputs
                n=1,
                stop=None
            ).choices[0].text.strip()
            
            # Calculate similarity score (using word overlap for simplicity)
            phi2_words = set(phi2_response.lower().split())
            wrn_words = set(wrn_response.lower().split())
            
            if not wrn_words:  # Avoid division by zero
                similarity = 0.0
            else:
                common_words = phi2_words & wrn_words
                similarity = len(common_words) / len(wrn_words)
            
            whiterabbitneo_scores.append(similarity)
            logging.info(f"Sample {i+1}: WhiteRabbitNeo similarity score: {similarity:.4f}")
            logging.info(f"  Prompt: {prompt[:50]}...")
            logging.info(f"  Phi-2: {phi2_response[:50]}...")
            logging.info(f"  WhiteRabbitNeo: {wrn_response[:50]}...")
            
        except Exception as e:
            logging.error(f"Error in WhiteRabbitNeo evaluation for prompt {i+1}: {e}")
    
    if not whiterabbitneo_scores:
        return 0.0
        
    avg_score = sum(whiterabbitneo_scores) / len(whiterabbitneo_scores)
    logging.info(f"Average WhiteRabbitNeo similarity score: {avg_score:.4f}")
    return avg_score

# --------------------------------------------------------------------------
# Training loop with early stopping and checkpointing
# --------------------------------------------------------------------------
best_val_loss = float("inf")
patience = 8  # Increased from 4 to 8 for more training time
no_improvement_count = 0

try:
    logging.info("Starting training loop...")
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            try:
                # Forward pass
                outputs = student_model(**batch)
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    accelerator.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                completed_steps += 1
                if completed_steps % 5 == 0:
                    gpu_memory = (torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0
                    logging.info(f"Step: {completed_steps} | Loss: {loss.item():.4f} | GPU Memory: {gpu_memory:.1f}MB")
            
            except Exception as e:
                logging.error(f"Error during training step {step}: {e}")
                continue
        
        # Evaluate after each epoch
        val_loss = evaluate(student_model, val_dataloader)
        logging.info(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")
        
        # Every epoch, also evaluate with WhiteRabbitNeo comparison (increased from every other epoch)
        wrn_similarity = evaluate_with_whiterabbitneo(tokenizer, student_model, val_teacher_pairs)
        logging.info(f"Epoch {epoch+1} WhiteRabbitNeo similarity score: {wrn_similarity:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            
            # Save the best checkpoint
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(student_model)
            best_ckpt_path = "./distilled_model_phi2_improved/best_checkpoint"
            try:
                unwrapped_model.save_pretrained(
                    best_ckpt_path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save
                )
                logging.info(f"Best checkpoint updated at epoch {epoch+1}, val_loss={val_loss:.4f}")
            except Exception as e:
                logging.error(f"Error saving best checkpoint: {e}")
        else:
            no_improvement_count += 1
            logging.info(f"No improvement. Patience counter: {no_improvement_count}/{patience}")
            if no_improvement_count >= patience:
                logging.info("Early stopping triggered.")
                break
    
    logging.info("Training completed (or early stopped). Saving the final model...")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(student_model)
    try:
        unwrapped_model.save_pretrained(
            "./distilled_model_phi2_improved/final",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        logging.info("Final model saved successfully")
        
        # Create a README file for the distilled model
        readme_content = f"""# Improved Distilled Phi-2 Model with WhiteRabbitNeo Evaluation

This model was created by distilling knowledge from Phi-4 into Phi-2 using LoRA adapters,
with additional evaluation and data augmentation from WhiteRabbitNeo. This is an improved
version with more training epochs, better prompting, and enhanced parameters.

## Training Details
- Primary Teacher Model: Phi-4 (via vLLM API)
- Secondary Teacher Model: WhiteRabbitNeo (via vLLM API)
- Student Model: Phi-2 with LoRA adapters
- Training Date: {datetime.now().strftime('%Y-%m-%d')}
- Best Validation Loss: {best_val_loss:.4f}
- Training Approach: Ensemble distillation with WhiteRabbitNeo evaluation
- Improvements:
  - Increased training epochs (15 instead of 8)
  - Enhanced prompt templates for technical requirements extraction
  - More domain-specific examples (20 instead of 10)
  - Lower temperature (0.3 instead of 0.7) for more focused outputs
  - Higher LoRA rank (32 instead of 16) for better learning capacity
  - Longer sequence length (256 instead of 128) for more context

## Usage
This model can be loaded with the Hugging Face transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Load the adapter
model = PeftModel.from_pretrained(base_model, "path/to/this/adapter")

# Generate text
prompt = "Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\\n\\nThe Cisco Catalyst 9300 Series switches..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
        
        with open("./distilled_model_phi2_improved/README.md", "w") as f:
            f.write(readme_content)
        
        logging.info("README file created for the distilled model")
    except Exception as e:
        logging.error(f"Error in final save: {e}")

except Exception as e:
    logging.error(f"Fatal error during training: {e}")
    raise
