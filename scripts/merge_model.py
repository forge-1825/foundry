from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("merge_model.log", mode='w')
    ]
)

def merge_model():
    """
    Merge the base Phi-2 model with the trained LoRA adapters.
    This creates a standalone model that can be served with vLLM.
    """
    logging.info("Starting model merging process...")
    
    # Check if the trained model exists
    adapter_path = "./distilled_model_phi2/best_checkpoint"
    if not os.path.exists(adapter_path):
        logging.error(f"Adapter path not found: {adapter_path}")
        logging.info("Checking for final model instead...")
        adapter_path = "./distilled_model_phi2/final"
        if not os.path.exists(adapter_path):
            logging.error(f"Final model path not found either: {adapter_path}")
            logging.error("Please ensure the distillation process completed successfully.")
            return False
    
    logging.info(f"Using adapter from: {adapter_path}")
    
    try:
        # Load the base model
        logging.info("Loading base Phi-2 model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        logging.info("Base model loaded successfully.")
        
        # Load the adapter
        logging.info("Loading trained adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        logging.info("Adapter loaded successfully.")
        
        # Merge the adapter with the base model
        logging.info("Merging adapter with base model...")
        merged_model = model.merge_and_unload()
        logging.info("Models merged successfully.")
        
        # Save the merged model
        output_path = "./merged_distilled_phi2"
        logging.info(f"Saving merged model to {output_path}...")
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logging.info("Merged model saved successfully.")
        
        return True
    
    except Exception as e:
        logging.error(f"Error during model merging: {e}")
        return False

if __name__ == "__main__":
    success = merge_model()
    if success:
        logging.info("Model merging completed successfully.")
        print("\nModel merging completed successfully.")
        print("To start the vLLM server with your merged model, run:")
        print("python -m vllm.entrypoints.openai.api_server --model ./merged_distilled_phi2 --port 8002")
        print("\nThen you can query it with curl:")
        print('''curl http://localhost:8002/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "merged_distilled_phi2",
    "prompt": "Extract detailed technical requirements from the following device specification:\\n\\nThe Cisco Catalyst 9300 Series switches...",
    "max_tokens": 200,
    "temperature": 0.7
  }'
''')
    else:
        logging.error("Model merging failed. Check the logs for details.")
        print("\nModel merging failed. Check the logs for details.")
