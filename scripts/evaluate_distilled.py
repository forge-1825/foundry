from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation.log")
    ]
)

def load_model(model_path, model_name=None, is_distilled=False):
    """Load a model and tokenizer."""
    try:
        if is_distilled:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def generate_text(model, tokenizer, prompt, max_length=100):
    """Generate text from a prompt."""
    try:
        # Encode the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Record start time
        start_time = time.time()
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.to(model.device),
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Record end time
        end_time = time.time()
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text, end_time - start_time
    except Exception as e:
        logging.error(f"Error generating text: {str(e)}")
        raise

def main():
    # Model paths and names
    original_model_name = "WhiteRabbitNeo/WhiteRabbitNeo-13B-v1"
    distilled_model_path = "./distilled_model_final"
    
    # Test prompts
    test_prompts = [
        "What are the key features of RouterOS?",
        "Explain how to configure a MikroTik router.",
        "What is the purpose of the CCR series routers?",
        "How do I set up a basic firewall in RouterOS?"
    ]
    
    results = {
        'original': {'times': [], 'outputs': []},
        'distilled': {'times': [], 'outputs': []}
    }
    
    try:
        # Load original model
        logging.info("Loading original model...")
        original_model, original_tokenizer = load_model(None, original_model_name, False)
        logging.info("Original model loaded successfully")
        
        # Load distilled model
        logging.info("Loading distilled model...")
        distilled_model, distilled_tokenizer = load_model(distilled_model_path, original_model_name, True)
        logging.info("Distilled model loaded successfully")
        
        # Test both models
        for i, prompt in enumerate(test_prompts, 1):
            logging.info(f"\nTesting prompt {i}: {prompt[:50]}...")
            
            # Test original model
            logging.info("Generating with original model...")
            original_output, original_time = generate_text(original_model, original_tokenizer, prompt)
            results['original']['times'].append(original_time)
            results['original']['outputs'].append(original_output)
            logging.info(f"Original model time: {original_time:.2f}s")
            
            # Test distilled model
            logging.info("Generating with distilled model...")
            distilled_output, distilled_time = generate_text(distilled_model, distilled_tokenizer, prompt)
            results['distilled']['times'].append(distilled_time)
            results['distilled']['outputs'].append(distilled_output)
            logging.info(f"Distilled model time: {distilled_time:.2f}s")
            
            # Log outputs
            logging.info("\nOriginal output:")
            logging.info(original_output)
            logging.info("\nDistilled output:")
            logging.info(distilled_output)
        
        # Calculate and log statistics
        avg_original_time = sum(results['original']['times']) / len(results['original']['times'])
        avg_distilled_time = sum(results['distilled']['times']) / len(results['distilled']['times'])
        speedup = avg_original_time / avg_distilled_time if avg_distilled_time > 0 else 0
        
        logging.info("\nEvaluation Results:")
        logging.info(f"Average original model time: {avg_original_time:.2f}s")
        logging.info(f"Average distilled model time: {avg_distilled_time:.2f}s")
        logging.info(f"Speedup factor: {speedup:.2f}x")
        
        # Save results
        results['statistics'] = {
            'avg_original_time': avg_original_time,
            'avg_distilled_time': avg_distilled_time,
            'speedup_factor': speedup,
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logging.info("Results saved to evaluation_results.json")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
