#!/usr/bin/env python3
"""
Simple script to query a model with a prompt and return the response.
This is used by the model query API endpoint.
"""

import argparse
import os
import sys
import logging
import json
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

def query_model(model_path, prompt):
    """
    Query a model with a prompt and return the response.
    This is a simple implementation that uses the model's CLI interface.
    """
    try:
        # Check if model path exists
        if not os.path.exists(model_path):
            logging.error(f"Model path does not exist: {model_path}")
            return f"Error: Model path does not exist: {model_path}"
        
        # For demonstration, we'll use a simple curl command to query the model
        # In a real implementation, you would use the appropriate library for your model
        # For example, for Hugging Face models, you would use the transformers library
        
        # Example curl command for a local API endpoint
        # This is just a placeholder - replace with your actual model query logic
        cmd = [
            "curl", "-s", "-X", "POST", 
            "http://localhost:8000/v1/completions",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "model": os.path.basename(model_path),
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7
            })
        ]
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Error querying model: {result.stderr}")
            return f"Error querying model: {result.stderr}"
        
        # Parse the response
        try:
            response_json = json.loads(result.stdout)
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["text"]
            else:
                return result.stdout
        except json.JSONDecodeError:
            # If the response is not valid JSON, return the raw output
            return result.stdout
    
    except Exception as e:
        logging.error(f"Error querying model: {str(e)}")
        return f"Error querying model: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Query a model with a prompt")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the model")
    
    args = parser.parse_args()
    
    response = query_model(args.model_path, args.prompt)
    print(response)

if __name__ == "__main__":
    main()
