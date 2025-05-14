import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Force offline mode

# Define the parent model and student model paths using raw strings
parent_model_path = r"G:\vllm-workspace\whiterabbitneo_parent"  # This is your phi4 parent model
student_model_path = r"G:\vllm-workspace\whiterabbitneo_vllm"

# Check if the paths exist
print("Parent model exists:", os.path.exists(parent_model_path))
print("Student model exists:", os.path.exists(student_model_path))

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
import torch.nn.functional as F
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Union

# --------------------------------------------------------------------------
# Logging configuration: log both to console and to a file (overwriting old logs)
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", mode='w')
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
# Load the parent's (phi4) configuration which is known to work with vLLM
# --------------------------------------------------------------------------
logging.info("Loading parent model configuration from the parent model path...")
parent_config = AutoConfig.from_pretrained(parent_model_path, trust_remote_code=True)
logging.info("Parent configuration loaded successfully.")

# --------------------------------------------------------------------------
# Loading student model and tokenizer (offline mode)
# --------------------------------------------------------------------------
logging.info("Loading student model and tokenizer offline...")

# Load the tokenizer from the student model folder
tokenizer = AutoTokenizer.from_pretrained(student_model_path, local_files_only=True)

# Clear CUDA cache if using GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    logging.info("Cleared CUDA cache")

# Create an offload folder for any weights that must be stored on disk
offload_folder = "./offload"
os.makedirs(offload_folder, exist_ok=True)

# Configure 4-bit quantization settings via bitsandbytes.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # or torch.float16 if preferred
)

# Load the student model with quantization and offload support,
# but override its config with the parent's config.
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_path,
    config=parent_config,  # Inherit configuration from the parent (phi4)
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder=offload_folder,
    torch_dtype=torch.bfloat16,  # Using BF16 on GPU for efficiency
    local_files_only=True         # Ensure offline mode is used
)

student_model.gradient_checkpointing_enable()
student_model.config.use_cache = False

print("Student model loaded successfully on", next(student_model.parameters()).device)
if torch.cuda.is_available():
    logging.info(f"Model device: {next(student_model.parameters()).device}")
    logging.info(f"Initial GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
logging.info(f"Student model loaded successfully from {student_model_path}.")

# --------------------------------------------------------------------------
# Load teacher pairs (the distilled teacher outputs)
# --------------------------------------------------------------------------
input_teacher_pairs = r"G:\NeedInput\Output\teacher_pairs.json"
logging.info(f"Loading teacher pairs from {input_teacher_pairs}...")
with open(input_teacher_pairs, "r", encoding="utf-8") as f:
    teacher_pairs = json.load(f)
logging.info(f"Loaded {len(teacher_pairs)} teacher pairs.")

# --------------------------------------------------------------------------
# Load teacher model for contrastive distillation from local snapshot.
# --------------------------------------------------------------------------
teacher_model_path = "/vllm-workspace/phi4_local_model"
teacher_offload_folder = "./teacher_offload"
os.makedirs(teacher_offload_folder, exist_ok=True)

from transformers import AutoModelForCausalLM
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_path,
    local_files_only=True,            # Use only local files
    repo_type="model",                # Specify that this is a model repository
    device_map={"": "cpu"},           # Keep teacher on CPU
    offload_folder=teacher_offload_folder,
    torch_dtype=torch.float16,        # Use FP16 for teacher (can be adjusted)
    trust_remote_code=True             # Allow custom code if required
)
teacher_model.eval()
logging.info(f"Teacher model loaded from {teacher_model_path} on CPU with offload support.")

# --------------------------------------------------------------------------
# Split teacher pairs into train and validation sets for early stopping
# --------------------------------------------------------------------------
val_size = min(50, int(0.1 * len(teacher_pairs)))  # e.g., 10% or up to 50 examples
train_teacher_pairs = teacher_pairs[:-val_size]
val_teacher_pairs = teacher_pairs[-val_size:]
logging.info(f"Training set size: {len(train_teacher_pairs)}")
logging.info(f"Validation set size: {len(val_teacher_pairs)}")

# --------------------------------------------------------------------------
# Define data collator and custom dataset for distillation training.
# We use a very short max sequence length (8 tokens) for testing/memory optimization.
# You can later increase this length.
# --------------------------------------------------------------------------
@dataclass
class DataCollatorForDistillation:
    tokenizer: AutoTokenizer
    max_length: int = 8  # For testing; adjust for real training
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

class DistillationDataset(torch.utils.data.Dataset):
    def __init__(self, teacher_pairs, tokenizer, max_length=8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logging.info("Tokenizer settings:")
        logging.info(f"BOS token: {tokenizer.bos_token} ({tokenizer.bos_token_id})")
        logging.info(f"EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
        logging.info(f"PAD token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
        
        for pair in teacher_pairs:
            if pair.get("target") and len(pair["target"].strip()) > 0:
                target_text = pair["target"].strip()
                if target_text:
                    if len(self.data) == 0:
                        logging.info(f"Processing first example: {target_text[:100]}...")
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
                        logging.info(f"First example decoded: {self.tokenizer.decode(input_ids, skip_special_tokens=False)[:100]}...")
                    self.data.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask
                    })
        
        logging.info(f"Dataset created with {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = DistillationDataset(train_teacher_pairs, tokenizer)
val_dataset = DistillationDataset(val_teacher_pairs, tokenizer)

os.makedirs("./distilled_model", exist_ok=True)

# --------------------------------------------------------------------------
# Apply PEFT for memory-efficient training: prepare model for k-bit training and apply LoRA.
# --------------------------------------------------------------------------
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

logging.info("Preparing model for k-bit training...")
student_model = prepare_model_for_kbit_training(student_model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
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
accelerator = Accelerator(
    gradient_accumulation_steps=32,
    mixed_precision="bf16" if torch.cuda.is_available() else None
)

# Create data loaders.
train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=DataCollatorForDistillation(tokenizer=tokenizer)
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=DataCollatorForDistillation(tokenizer=tokenizer)
)

# Calculate total training steps.
num_epochs = 5  # Increase beyond 1 so we can see early stopping
num_training_steps = num_epochs * len(train_dataloader)

# Create optimizer and learning rate scheduler.
optimizer = AdamW(student_model.parameters(), lr=5e-5, eps=1e-4)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

# Prepare the model, optimizer, dataloaders, and scheduler with Accelerator.
student_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
    student_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
)

student_model.train()
completed_steps = 0

# --------------------------------------------------------------------------
# Define a contrastive loss function using in-batch negatives (InfoNCE-style)
# --------------------------------------------------------------------------
def contrastive_loss(student_emb, teacher_emb, temperature=0.07):
    # Normalize embeddings (L2 norm)
    student_emb = F.normalize(student_emb, p=2, dim=1)
    teacher_emb = F.normalize(teacher_emb, p=2, dim=1)
    # Compute similarity matrix [batch_size x batch_size]
    logits = torch.matmul(student_emb, teacher_emb.t()) / temperature
    # Each sample should match its own teacher embedding
    labels = torch.arange(student_emb.size(0), device=student_emb.device)
    return F.cross_entropy(logits, labels)

# --------------------------------------------------------------------------
# Define function to get teacher hidden states (for both contrastive and intermediate losses)
# --------------------------------------------------------------------------
def get_teacher_hidden_states(batch):
    with torch.no_grad():
        outputs = teacher_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True
        )
    return outputs.hidden_states

# --------------------------------------------------------------------------
# Define an intermediate loss function that compares an intermediate layer (e.g., second-to-last)
# --------------------------------------------------------------------------
def intermediate_loss(student_hidden_states, teacher_hidden_states, layer_index=-2):
    # Compare the [CLS]-like token (first token) from the chosen layer using MSE loss.
    return F.mse_loss(student_hidden_states[layer_index][:, 0, :], teacher_hidden_states[layer_index][:, 0, :])

# --------------------------------------------------------------------------
# Define a simple evaluation function for validation (computes average loss)
# --------------------------------------------------------------------------
def evaluate(model, dataloader):
    model.eval()
    losses = []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            loss = outputs.loss
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if len(losses) > 0 else float("inf")

# Hyperparameters for loss weighting
beta = 0.1               # Weight for contrastive loss
lambda_intermediate = 0.1  # Weight for intermediate (feature-based) loss

# --------------------------------------------------------------------------
# Training loop with early stopping, contrastive loss, intermediate loss, and checkpointing
# --------------------------------------------------------------------------
best_val_loss = float("inf")
patience = 2
no_improvement_count = 0

try:
    logging.info("Starting training loop...")
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(train_dataloader):
            try:
                # Forward pass of the student model (with hidden states)
                outputs = student_model(**batch, output_hidden_states=True)
                loss_kd = outputs.loss  # The standard knowledge distillation (KD) loss
                
                # Get teacher hidden states for this batch (running on CPU with offload)
                teacher_hidden_states = get_teacher_hidden_states(batch)
                # Use the last hidden state (first token) for contrastive loss
                teacher_emb = teacher_hidden_states[-1][:, 0, :]
                
                # Compute contrastive loss between student and teacher embeddings
                student_emb = outputs.hidden_states[-1][:, 0, :]
                loss_contrast = contrastive_loss(student_emb, teacher_emb, temperature=0.07)
                
                # Compute intermediate loss on an earlier hidden layer (e.g., second-to-last)
                loss_intermediate = intermediate_loss(outputs.hidden_states, teacher_hidden_states, layer_index=-2)
                
                # Combine losses
                total_loss = loss_kd + beta * loss_contrast + lambda_intermediate * loss_intermediate
                accelerator.backward(total_loss)
                
                if (step + 1) % accelerator.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                completed_steps += 1
                if completed_steps % 1 == 0:
                    gpu_memory = (torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0
                    logging.info(
                        f"Step: {completed_steps} | KD Loss: {loss_kd.item():.4f} | Contrast Loss: {loss_contrast.item():.4f} | Intermediate Loss: {loss_intermediate.item():.4f} | Total Loss: {total_loss.item():.4f} | GPU Memory: {gpu_memory:.1f}MB"
                    )
                
                if completed_steps % 5 == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(student_model)
                    checkpoint_path = f"./distilled_model/checkpoint-{completed_steps}"
                    try:
                        unwrapped_model.save_pretrained(
                            checkpoint_path,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save
                        )
                        logging.info(f"Checkpoint saved to {checkpoint_path}")
                    except Exception as e:
                        logging.error(f"Error saving checkpoint: {e}")
            
            except Exception as e:
                logging.error(f"Error during training step {step}: {e}")
                continue

        val_loss = evaluate(student_model, val_dataloader)
        logging.info(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(student_model)
            best_ckpt_path = "./distilled_model/best_checkpoint"
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
            "./distilled_model_final",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )
        logging.info("Final model saved successfully")
        
        metrics_file = os.path.join("./distilled_model", "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(accelerator.state.__dict__, f, indent=2, default=lambda o: str(o))
        logging.info(f"Training metrics saved to {metrics_file}")
    except Exception as e:
        logging.error(f"Error in final save: {e}")

except Exception as e:
    logging.error(f"Fatal error during training: {e}")
    raise
