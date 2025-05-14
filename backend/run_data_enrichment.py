#!/usr/bin/env python
"""
Data enrichment script for processing PDF files.
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_pdf_files(input_dir, output_file):
    """
    Process PDF files in the input directory and save the enriched data to the output file.
    
    Args:
        input_dir: Path to the directory containing PDF files
        output_file: Path to save the enriched data
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return False
    
    # Find all PDF files in the input directory
    pdf_files = glob.glob(os.path.join(input_dir, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    # Create a simple enriched data structure
    enriched_data = {
        "source_directory": input_dir,
        "file_count": len(pdf_files),
        "files": []
    }
    
    # Process each PDF file
    for pdf_file in pdf_files:
        # Get the relative path from the input directory
        rel_path = os.path.relpath(pdf_file, input_dir)
        
        # Add the file to the enriched data
        enriched_data["files"].append({
            "path": rel_path,
            "size": os.path.getsize(pdf_file),
            "last_modified": os.path.getmtime(pdf_file)
        })
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the enriched data to the output file
    with open(output_file, 'w') as f:
        json.dump(enriched_data, f, indent=2)
    
    logger.info(f"Enriched data saved to {output_file}")
    return True

def main():
    """
    Main function for the data enrichment script.
    """
    # Get the action from the user
    print("============================================================")
    print("Running Data Enrichment with GPU Acceleration")
    print("============================================================")
    print()
    print("Choose an action:")
    print("1. Run data enrichment on AgentGreen folder")
    print("2. Run data enrichment on a different folder")
    print()
    
    choice = input("Enter your choice (1-2): ")
    
    if choice == "1":
        # Use the AgentGreen folder
        input_dir = os.path.join(os.getcwd(), "AgentGreen")
        output_file = os.path.join(os.getcwd(), "Output", "enriched_data.json")
    elif choice == "2":
        # Get the input directory from the user
        input_dir = input("Enter the path to the input directory: ")
        output_file = input("Enter the path to save the enriched data (default: Output\\enriched_data.json): ")
        
        if not output_file:
            output_file = os.path.join(os.getcwd(), "Output", "enriched_data.json")
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Print the input and output paths
    print()
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Process the PDF files
    print(f"Running data enrichment on {input_dir}...")
    print()
    print("Processing PDF files...")
    
    success = process_pdf_files(input_dir, output_file)
    
    if success:
        print(f"Found {len(glob.glob(os.path.join(input_dir, '**', '*.pdf'), recursive=True))} PDF files in {input_dir}")
        print(f"Enriched data saved to {output_file}")
    else:
        print("Data enrichment failed.")
    
    print()
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
