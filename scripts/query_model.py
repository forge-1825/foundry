import os
import json
import logging
import torch
import argparse
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("query_model.log", mode='w')
    ]
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path):
    """
    Load the model from the specified path.
    
    Args:
        model_path: Path to the model (merged model or adapter)
    
    Returns:
        Tuple of (tokenizer, model)
    """
    global model, tokenizer, device
    
    logging.info(f"Loading model from {model_path}...")
    
    try:
        # First try to load as a merged model
        if os.path.exists(model_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                model = model.to(device)
                logging.info(f"Merged model loaded successfully and moved to {device}.")
                return True
            except Exception as e:
                logging.error(f"Error loading merged model: {e}")
                logging.info("Trying to load from adapter...")
        
        # If that fails, try to load from adapter
        try:
            base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
            
            # Check different possible adapter paths
            adapter_paths = [
                model_path,
                os.path.join(model_path, "best_checkpoint"),
                os.path.join(model_path, "final"),
                "./distilled_model_phi2/best_checkpoint",
                "./distilled_model_phi2/final",
                "./distilled_model_phi2_improved/best_checkpoint",
                "./distilled_model_phi2_improved/final"
            ]
            
            adapter_loaded = False
            for adapter_path in adapter_paths:
                if os.path.exists(adapter_path):
                    try:
                        model = PeftModel.from_pretrained(base_model, adapter_path)
                        model = model.to(device)
                        logging.info(f"Model loaded successfully from adapter at {adapter_path} and moved to {device}.")
                        adapter_loaded = True
                        break
                    except Exception as e:
                        logging.error(f"Error loading adapter from {adapter_path}: {e}")
            
            if not adapter_loaded:
                logging.error("Could not load model from any adapter path.")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading model from adapter: {e}")
            return False
    
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return False

def generate_response(prompt, temperature=0.3, max_new_tokens=200):
    """
    Generate a response from the model for the given prompt.
    
    Args:
        prompt: The input prompt
        temperature: Temperature for generation (lower = more focused)
        max_new_tokens: Maximum number of new tokens to generate
    
    Returns:
        The generated text
    """
    global model, tokenizer, device
    
    if model is None or tokenizer is None:
        return "Error: Model not loaded. Please load the model first."
    
    try:
        # Format the prompt with the technical requirements extraction template
        formatted_prompt = f"Extract detailed technical requirements from the following device specification. Format your response as a numbered list of requirements:\n\n{prompt}"
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text if it's included
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
        
        return generated_text
    
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

def interactive_mode():
    """
    Run the model in interactive mode, allowing the user to input prompts.
    """
    print("\n=== Interactive Mode ===")
    print("Enter your device specifications below. Type 'exit' to quit.")
    print("The model will extract technical requirements from your input.")
    
    while True:
        # Get user input
        user_input = input("\nEnter device specification (or 'exit' to quit): ")
        
        # Check if user wants to exit
        if user_input.lower() == 'exit':
            print("Exiting interactive mode.")
            break
        
        # Generate and print response
        print("\nGenerating response...")
        response = generate_response(user_input)
        print("\n=== Generated Technical Requirements ===\n")
        print(response)
        print("\n" + "=" * 80)

# Initialize Flask app for API mode
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query_endpoint():
    """
    API endpoint for querying the model.
    
    Expected JSON payload:
    {
        "prompt": "The device specification text",
        "temperature": 0.3,  # optional
        "max_new_tokens": 200  # optional
    }
    """
    try:
        # Get request data
        data = request.json
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request data"}), 400
        
        # Extract parameters
        prompt = data['prompt']
        temperature = data.get('temperature', 0.3)
        max_new_tokens = data.get('max_new_tokens', 200)
        
        # Generate response
        response = generate_response(prompt, temperature, max_new_tokens)
        
        # Return response
        return jsonify({
            "prompt": prompt,
            "response": response,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens
            }
        })
    
    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

def api_mode(host='0.0.0.0', port=5000):
    """
    Run the model in API mode, exposing an endpoint that can be queried with curl.
    """
    print(f"\n=== API Mode ===")
    print(f"Starting server on {host}:{port}")
    print("You can query the model using curl:")
    print(f"""
curl -X POST http://{host}:{port}/query \\
  -H "Content-Type: application/json" \\
  -d '{{
    "prompt": "The Cisco Catalyst 9300 Series switches are enterprise-class access switches that provide full convergence between wired and wireless networks.",
    "temperature": 0.3,
    "max_new_tokens": 200
  }}'
""")
    
    # Run the Flask app
    app.run(host=host, port=port)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query the trained model interactively or via API")
    parser.add_argument("--model_path", type=str, default="./merged_distilled_phi2", 
                        help="Path to the model (default: ./merged_distilled_phi2)")
    parser.add_argument("--mode", type=str, choices=["interactive", "api"], default="interactive",
                        help="Mode to run the script in (default: interactive)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the API server on (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the API server on (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    # Load the model
    if not load_model(args.model_path):
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Run in the specified mode
    if args.mode == "interactive":
        interactive_mode()
    else:  # api mode
        api_mode(args.host, args.port)
