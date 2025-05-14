#!/usr/bin/env python3
"""
Create mock datasets for OpenEvals testing.

This script generates sample datasets in JSONL format for different evaluation tasks.
"""

import json
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
DATASETS_DIR = os.path.join(os.getcwd(), "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

def create_error_suggestion_dataset():
    """Create a mock dataset for error suggestion evaluation."""
    examples = [
        {
            "input": "I'm getting a 'permission denied' error when trying to run my script",
            "reference": "This error occurs because your script doesn't have executable permissions. Run 'chmod +x script.sh' to add execute permission to the file."
        },
        {
            "input": "My Python program crashes with 'ImportError: No module named requests'",
            "reference": "This error means the 'requests' module is not installed. Install it using 'pip install requests'."
        },
        {
            "input": "Getting 'Address already in use' when starting my server",
            "reference": "This error occurs when another process is already using the port you're trying to bind to. Find the process using 'lsof -i :<port>' and either kill it or use a different port."
        },
        {
            "input": "I'm getting 'Connection refused' when trying to connect to my database",
            "reference": "This error indicates that either the database server is not running, a firewall is blocking the connection, or you're using the wrong host/port. Check if the database is running and accessible."
        },
        {
            "input": "My JavaScript code throws 'Uncaught TypeError: Cannot read property 'length' of undefined'",
            "reference": "This error occurs when you're trying to access a property of an undefined value. Make sure the object exists before accessing its properties, or use optional chaining (obj?.length)."
        }
    ]
    
    # Write examples to JSONL file
    dataset_path = os.path.join(DATASETS_DIR, "error_suggestion.jsonl")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Created error suggestion dataset with {len(examples)} examples at {dataset_path}")

def create_command_extraction_dataset():
    """Create a mock dataset for command extraction evaluation."""
    examples = [
        {
            "input": "How do I list all files in a directory including hidden ones?",
            "reference": "ls -la"
        },
        {
            "input": "I need to find all Python files containing the word 'import'",
            "reference": "find . -name \"*.py\" -exec grep -l \"import\" {} \\;"
        },
        {
            "input": "How can I check disk space usage?",
            "reference": "df -h"
        },
        {
            "input": "How do I recursively change permissions for all files in a directory?",
            "reference": "chmod -R 755 directory_name"
        },
        {
            "input": "I want to see the last 100 lines of a log file and follow new additions",
            "reference": "tail -n 100 -f logfile.log"
        }
    ]
    
    # Write examples to JSONL file
    dataset_path = os.path.join(DATASETS_DIR, "command_extraction.jsonl")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Created command extraction dataset with {len(examples)} examples at {dataset_path}")

def create_qa_dataset():
    """Create a mock dataset for QA evaluation."""
    examples = [
        {
            "input": "What is machine learning?",
            "reference": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
        },
        {
            "input": "Explain how a neural network works",
            "reference": "A neural network is a computational model inspired by the human brain. It consists of layers of interconnected nodes (neurons) that process information and learn patterns from data."
        },
        {
            "input": "What is the difference between supervised and unsupervised learning?",
            "reference": "Supervised learning uses labeled data where the model learns to predict outputs based on inputs, while unsupervised learning works with unlabeled data to find patterns or structure without specific output targets."
        },
        {
            "input": "What is transfer learning?",
            "reference": "Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task, leveraging knowledge gained from the first task to improve performance on the second."
        },
        {
            "input": "What is the purpose of regularization in machine learning?",
            "reference": "Regularization helps prevent overfitting by adding a penalty term to the loss function, discouraging complex models and promoting simpler ones that generalize better to unseen data."
        }
    ]
    
    # Write examples to JSONL file
    dataset_path = os.path.join(DATASETS_DIR, "qa.jsonl")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Created QA dataset with {len(examples)} examples at {dataset_path}")

def main():
    """Main entry point for the script."""
    logger.info("Creating mock datasets for OpenEvals testing")
    
    create_error_suggestion_dataset()
    create_command_extraction_dataset()
    create_qa_dataset()
    
    logger.info("All datasets created successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
