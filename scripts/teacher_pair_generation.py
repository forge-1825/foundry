import json
import os
import sys
import subprocess
import time
import logging
from tqdm import tqdm

# Configure logging to output messages with timestamps.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def generate_teacher_output(prompt, model_name="deepseek-r1:70b"):
    """
    Query the teacher model via Ollama with the given prompt.
    Returns the teacher model's output as a string.
    """
    try:
        # Build the command using 'run' instead of 'query'
        command = ["ollama", "run", model_name, prompt]
        logging.info(f"Sending prompt to teacher model {model_name}...")
        # Set encoding to UTF-8 and add a timeout
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60  # 60 second timeout
        )
        if result.returncode != 0:
            logging.error("Error querying teacher model: %s", result.stderr.strip())
            return ""
        logging.info("Teacher model returned output successfully.")
        return result.stdout.strip()
    except Exception as e:
        logging.error("Exception during teacher query: %s", e)
        return ""

def main():
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate teacher pairs for distillation")
    parser.add_argument("--input-file", type=str, help="Path to the enriched data JSON file")
    parser.add_argument("--output-file", type=str, help="Path to save the teacher pairs JSON file")
    parser.add_argument("--teacher-model", type=str, default="deepseek-r1:70b", help="Name of the teacher model to use")

    args = parser.parse_args()

    logging.info("=== Teacher Pair Generation Script Started ===")

    # Use command-line arguments or default paths
    input_path = args.input_file or r"G:\NeedInput\Output\enriched_data.json"
    output_path = args.output_file or r"G:\NeedInput\Output\teacher_pairs.json"

    logging.info("Using input file: %s", input_path)
    if not os.path.exists(input_path):
        logging.error("Input file not found: %s. Exiting.", input_path)
        sys.exit(1)

    # Load enriched dataset
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

    # Define a prompt template for teacher output generation.
    prompt_template = (
        "Extract detailed technical requirements from the following device specification:\n\n"
        "{input_text}\n\n"
        "Requirements:"
    )

    teacher_pairs = []
    logging.info("Beginning teacher output generation for each record...")

    for idx, record in enumerate(tqdm(dataset, desc="Processing records"), start=1):
        input_text = record.get("summary", "") or record.get("cleaned_text", "") or record.get("raw_content", "")
        if not input_text:
            logging.warning("Record %d (URL: %s) has no input text. Skipping.", idx, record.get("url", "N/A"))
            continue

        prompt = prompt_template.format(input_text=input_text)
        logging.info("Record %d: Generated prompt for URL %s", idx, record.get("url", "N/A"))

        teacher_model = args.teacher_model
        teacher_output = generate_teacher_output(prompt, teacher_model)
        if not teacher_output:
            logging.warning("Record %d (URL: %s) returned no teacher output.", idx, record.get("url", "N/A"))
        else:
            logging.info("Record %d (URL: %s): Teacher output generated using model %s.", idx, record.get("url", "N/A"), teacher_model)

        teacher_pairs.append({
            "url": record.get("url"),
            "input": input_text,
            "target": teacher_output
        })
        time.sleep(1)  # Increased delay between requests

    # Ensure output directory exists.
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception as e:
        logging.error("Error creating output directory: %s", e)
        sys.exit(1)

    # Save the teacher pairs as JSON.
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(teacher_pairs, f, ensure_ascii=False, indent=4)
        logging.info("Teacher pairs successfully saved to %s.", output_path)
    except Exception as e:
        logging.error("Error saving teacher pairs: %s", e)

    logging.info("=== Teacher Pair Generation Script Completed ===")

if __name__ == "__main__":
    main()
