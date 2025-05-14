import json
import os
import sys
import time
import logging
import subprocess
import argparse
from tqdm import tqdm
from openai import OpenAI

# Configure logging to output messages with timestamps.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate teacher pairs using vLLM.')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the input JSON file with enriched data')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Path to save the output teacher pairs JSON file')
    parser.add_argument('--teacher-model', type=str, default="jakiAJK/microsoft-phi-4_GPTQ-int4",
                        help='Name of the teacher model to use')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    logging.info("=== Teacher Pair Generation Script (vLLM) Started ===")
    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"Teacher model: {args.teacher_model}")
    
    # Configure OpenAI client to point to the locally running vLLM server
    client = OpenAI(
        base_url="http://localhost:8000/v1",  # Make sure your teacher model container is accessible at this endpoint
        api_key="not-needed"  # vLLM doesn't require an API key
    )
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logging.error(f"Input file not found: {args.input_file}. Exiting.")
        sys.exit(1)
    
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        logging.info(f"Loaded {len(dataset)} records from {args.input_file}.")
    except Exception as e:
        logging.error(f"Error loading JSON file: {e}")
        sys.exit(1)
    
    if not dataset:
        logging.error("Dataset is empty. Exiting.")
        sys.exit(1)
    
    # Define a prompt template for teacher output generation.
    prompt_template = (
        "Extract detailed technical requirements from the following device specification:\n\n"
        "{input_text}\n\n"
        "Requirements:"
    )
    
    teacher_pairs = []
    max_prompt_length = 1000  # Limit prompt length to avoid overwhelming the teacher model
    
    logging.info("Beginning teacher output generation for each record...")
    
    try:
        for idx, record in enumerate(tqdm(dataset, desc="Processing records"), start=1):
            input_text = record.get("summary", "") or record.get("cleaned_text", "") or record.get("raw_content", "")
            if not input_text:
                logging.warning(f"Record {idx} (URL: {record.get('url', 'N/A')}) has no input text. Skipping.")
                continue

            # Truncate input_text if it's too long
            if len(input_text) > max_prompt_length:
                logging.info(f"Truncating input text for record {idx} (URL: {record.get('url', 'N/A')})")
                input_text = input_text[:max_prompt_length]
            
            prompt = prompt_template.format(input_text=input_text)
            logging.info(f"Record {idx}: Generated prompt for URL {record.get('url', 'N/A')}")
            
            # Query the teacher model
            try:
                logging.info(f"Sending prompt to vLLM teacher model...")
                response = client.completions.create(
                    model=args.teacher_model,
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7,
                    n=1,
                    stop=None
                )
                teacher_output = response.choices[0].text.strip()
                logging.info("Teacher model returned output successfully.")
            except Exception as e:
                logging.error(f"Error querying vLLM teacher model: {e}")
                teacher_output = ""
            
            if not teacher_output:
                logging.warning(f"Record {idx} (URL: {record.get('url', 'N/A')}) returned no teacher output.")
            else:
                logging.info(f"Record {idx} (URL: {record.get('url', 'N/A')}): Teacher output generated.")
            
            teacher_pairs.append({
                "url": record.get("url"),
                "input": input_text,
                "target": teacher_output
            })
            time.sleep(0.5)  # Optional delay to avoid overloading the server
    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user. Saving progress...")
        # Continue to save what we have so far
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating output directory: {e}")
            sys.exit(1)
    
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(teacher_pairs, f, ensure_ascii=False, indent=4)
        logging.info(f"Teacher pairs successfully saved to {args.output_file}.")
    except Exception as e:
        logging.error(f"Error saving teacher pairs: {e}")
    
    logging.info("=== Teacher Pair Generation Script (vLLM) Completed ===")

if __name__ == "__main__":
    main()
