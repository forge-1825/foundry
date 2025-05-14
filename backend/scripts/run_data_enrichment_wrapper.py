#!/usr/bin/env python
"""
Wrapper script for run_data_enrichment.bat that automates the input process.
This script takes command-line arguments and passes them to the batch file.
"""

import os
import sys
import subprocess
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_data_enrichment(source_folder, output_dir=None):
    """
    Run the data enrichment batch file with the given parameters.

    Args:
        source_folder: Path to the folder containing the PDF files
        output_dir: Path to save the enriched data (optional)

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    try:
        # Find the batch file
        possible_paths = [
            os.path.join('..', '..', 'run_data_enrichment.bat'),  # From backend/scripts to root
            os.path.join('..', '..', '..', 'run_data_enrichment.bat'),  # From backend/scripts to parent of root
            os.path.join('..', '..', 'scripts', 'run_data_enrichment.bat'),  # Check in scripts directory
        ]

        script_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                script_path = abs_path
                logger.info(f"Found script at: {script_path}")
                break

        if not script_path:
            logger.error("Could not find run_data_enrichment.bat")
            return 1, "", "Could not find run_data_enrichment.bat"

        # Instead of using the batch file, let's directly call the Python script
        # Find the Python script
        python_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'run_data_enrichment.py')

        if not os.path.exists(python_script_path):
            logger.error(f"Could not find run_data_enrichment.py at {python_script_path}")
            return 1, "", f"Could not find run_data_enrichment.py at {python_script_path}"

        logger.info(f"Found Python script at: {python_script_path}")

        # Create a temporary input file for the Python script
        temp_input_file = os.path.join(os.path.dirname(python_script_path), "temp_input.txt")

        with open(temp_input_file, "w") as f:
            f.write(f"2\n")
            f.write(f"{source_folder}\n")
            if output_dir:
                f.write(f"{output_dir}\n")
            else:
                f.write(f"\n")
            f.write(f"\n")

        logger.info(f"Created temporary input file: {temp_input_file}")

        # Run the Python script with the input file
        try:
            process = subprocess.Popen(
                [sys.executable, python_script_path],
                stdin=open(temp_input_file, 'r'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for the process to complete
            stdout, stderr = process.communicate(timeout=120)  # Set a timeout of 120 seconds

            # Clean up the temporary input file
            try:
                os.remove(temp_input_file)
                logger.info(f"Removed temporary input file: {temp_input_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary input file: {e}")

            logger.info(f"Process completed with return code: {process.returncode}")
            logger.info(f"stdout: {stdout}")
            logger.info(f"stderr: {stderr}")

            return process.returncode, stdout, stderr

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            logger.error("Process timed out and was killed")

            # Clean up the temporary input file
            try:
                os.remove(temp_input_file)
                logger.info(f"Removed temporary input file: {temp_input_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary input file: {e}")

            return 1, stdout, "Process timed out and was killed"

    except Exception as e:
        logger.error(f"Error running data enrichment: {e}")
        return 1, "", str(e)

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_data_enrichment_wrapper.py <source_folder> [output_dir]")
        sys.exit(1)

    source_folder = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # Run the data enrichment
    return_code, stdout, stderr = run_data_enrichment(source_folder, output_dir)

    # Print the output
    print(stdout)

    # Exit with the same return code
    sys.exit(return_code)
