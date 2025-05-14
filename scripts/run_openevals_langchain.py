#!/usr/bin/env python3
"""
OpenEvals Integration Script for Model Distillation Pipeline

This script runs evaluations using LangChain evaluators and custom datasets
for tasks like 'Error Suggestion' and 'Command Extraction'.
"""

import argparse
import json
import os
import sys
import time
import yaml
from datetime import datetime
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Mock imports for demonstration
try:
    import langchain
    from langchain.evaluation import load_evaluator
    from langchain.evaluation.schema import StringEvaluator
except ImportError:
    logger.warning("LangChain not installed. This script will run in mock mode.")
    langchain = None

# Constants
RESULTS_DIR = os.path.join(os.getcwd(), "results")
DATASETS_DIR = os.path.join(os.getcwd(), "datasets")
CONFIG_DIR = os.path.join(os.getcwd(), "configs")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Mock evaluator classes for demonstration
class MockStringEvaluator:
    """Mock evaluator for demonstration purposes."""
    
    def __init__(self, name="mock_evaluator"):
        self.name = name
    
    def evaluate_strings(self, prediction, reference):
        """Simulate evaluation with random scores."""
        time.sleep(0.1)  # Simulate processing time
        return {
            "score": round(random.uniform(0.5, 1.0), 4),
            "explanation": f"Mock evaluation of prediction against reference."
        }

class MockLLMEvaluator:
    """Mock LLM-as-judge evaluator."""
    
    def __init__(self, name="llm_judge"):
        self.name = name
    
    def evaluate_strings(self, prediction, reference, input_text=None):
        """Simulate LLM evaluation with random scores."""
        time.sleep(0.2)  # Simulate LLM processing time
        score = round(random.uniform(0.6, 0.95), 4)
        return {
            "score": score,
            "explanation": f"The response {'adequately' if score > 0.8 else 'partially'} addresses the query with {'comprehensive' if score > 0.9 else 'basic'} information."
        }

def load_dataset(dataset_path):
    """Load a dataset from a JSONL file."""
    examples = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        logger.info(f"Loaded {len(examples)} examples from {dataset_path}")
        return examples
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_path}: {e}")
        return []

def load_config(config_path):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return {}

def get_evaluator(evaluator_id, llm=None):
    """Get an evaluator by ID."""
    if langchain is None:
        # Return mock evaluators in demo mode
        if evaluator_id in ["error_suggestion_judge", "command_extraction_judge", "qa_correctness"]:
            return MockLLMEvaluator(name=evaluator_id)
        else:
            return MockStringEvaluator(name=evaluator_id)
    
    # In a real implementation, this would use LangChain's evaluator loading
    try:
        if evaluator_id == "string_match":
            return load_evaluator("string_distance")
        elif evaluator_id == "embedding_similarity":
            return load_evaluator("embedding_distance")
        elif evaluator_id in ["error_suggestion_judge", "command_extraction_judge", "qa_correctness"]:
            return load_evaluator("llm", llm=llm)
        else:
            logger.warning(f"Unknown evaluator {evaluator_id}, using string_distance as fallback")
            return load_evaluator("string_distance")
    except Exception as e:
        logger.error(f"Error loading evaluator {evaluator_id}: {e}")
        return MockStringEvaluator(name=evaluator_id)

def query_model(model_endpoint, prompt, model_id=None):
    """Query a model API endpoint."""
    # In a real implementation, this would make an API call
    # For demonstration, we'll return mock responses
    time.sleep(0.3)  # Simulate API latency
    
    if "error" in prompt.lower():
        return "This error typically occurs when there's a permission issue or configuration problem. Try checking the logs for more details and ensure all dependencies are installed correctly."
    elif "command" in prompt.lower():
        return "```\ngrep -r 'pattern' /path/to/directory\n```"
    else:
        return "I'm not sure about that specific question. Could you provide more details or context?"

def run_evaluation(config, dataset_type=None, model_id=None):
    """Run an evaluation using the specified configuration."""
    # Load configuration
    if not config:
        logger.error("No configuration provided")
        return False
    
    # Determine dataset to use
    if dataset_type:
        dataset_path = os.path.join(DATASETS_DIR, f"{dataset_type}.jsonl")
    elif "dataset" in config:
        dataset_path = os.path.join(DATASETS_DIR, f"{config['dataset']}.jsonl")
        dataset_type = config["dataset"]
    else:
        logger.error("No dataset specified")
        return False
    
    # Check if dataset exists, if not create a mock dataset
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset {dataset_path} not found, creating mock dataset")
        create_mock_dataset(dataset_path, dataset_type)
    
    # Load dataset
    examples = load_dataset(dataset_path)
    if not examples:
        logger.error("Failed to load dataset")
        return False
    
    # Determine models to evaluate
    models = config.get("models", {})
    if not models:
        logger.warning("No models specified in config, using defaults")
        models = {
            "teacher": {
                "name": "Llama 3 8B Instruct (Teacher)",
                "endpoint": "http://localhost:8000/v1",
                "model_id": "casperhansen/llama-3-8b-instruct-awq",
                "type": "teacher"
            },
            "student": {
                "name": "Phi-3 Mini (Student)",
                "endpoint": "http://localhost:8002/v1",
                "model_id": "microsoft/Phi-3-mini-4k-instruct",
                "type": "student"
            }
        }
    
    # Determine evaluators to use
    evaluators = config.get("evaluators", [])
    if not evaluators:
        logger.warning("No evaluators specified in config, using defaults")
        if dataset_type == "error_suggestion":
            evaluators = ["error_suggestion_judge", "string_match"]
        elif dataset_type == "command_extraction":
            evaluators = ["command_extraction_judge", "string_match"]
        else:
            evaluators = ["string_match", "embedding_similarity"]
    
    # Initialize evaluator objects
    evaluator_objects = {}
    for evaluator_id in evaluators:
        evaluator_objects[evaluator_id] = get_evaluator(evaluator_id)
    
    # Initialize results structure
    results = {
        "run_name": config.get("run_name", f"{dataset_type.capitalize()} Evaluation"),
        "timestamp": datetime.now().isoformat(),
        "dataset_type": dataset_type,
        "dataset_size": len(examples),
        "models": {},
        "evaluator_scores": {}
    }
    
    # Initialize model results
    for model_id, model_info in models.items():
        results["models"][model_id] = {
            "name": model_info.get("name", model_id),
            "endpoint": model_info.get("endpoint", "http://localhost:8000/v1"),
            "model_id": model_info.get("model_id", "unknown"),
            "type": model_info.get("type", "unknown"),
            "scores": {},
            "examples": []
        }
        
        # Initialize evaluator scores for this model
        for evaluator_id in evaluators:
            results["models"][model_id]["scores"][evaluator_id] = {
                "average": 0.0,
                "min": 1.0,
                "max": 0.0,
                "std": 0.0
            }
    
    # Initialize evaluator scores across models
    for evaluator_id in evaluators:
        results["evaluator_scores"][evaluator_id] = {}
        for model_id in models:
            results["evaluator_scores"][evaluator_id][model_id] = 0.0
    
    # Run evaluation for each model
    for model_id, model_info in models.items():
        logger.info(f"Evaluating model: {model_info.get('name', model_id)}")
        
        # Track scores for calculating statistics
        all_scores = {evaluator_id: [] for evaluator_id in evaluators}
        
        # Process examples
        for i, example in enumerate(examples):
            if i >= config.get("max_examples", 50):  # Limit number of examples for demo
                break
                
            logger.info(f"Evaluating example {i+1}/{min(len(examples), config.get('max_examples', 50))}")
            
            # Extract input and reference
            input_text = example.get("input", "")
            reference_output = example.get("reference", example.get("reference_output", ""))
            
            # Query model
            model_output = query_model(
                model_info.get("endpoint", "http://localhost:8000/v1"),
                input_text,
                model_info.get("model_id", "unknown")
            )
            
            # Evaluate output with each evaluator
            example_scores = {}
            for evaluator_id, evaluator in evaluator_objects.items():
                try:
                    if hasattr(evaluator, "evaluate_strings"):
                        if isinstance(evaluator, MockLLMEvaluator):
                            result = evaluator.evaluate_strings(model_output, reference_output, input_text)
                        else:
                            result = evaluator.evaluate_strings(model_output, reference_output)
                    else:
                        # Fallback for other evaluator interfaces
                        result = {"score": round(random.uniform(0.5, 1.0), 4)}
                    
                    score = result.get("score", 0.0)
                    all_scores[evaluator_id].append(score)
                    
                    example_scores[evaluator_id] = {
                        "score": score,
                        "explanation": result.get("explanation", "")
                    }
                except Exception as e:
                    logger.error(f"Error evaluating with {evaluator_id}: {e}")
                    example_scores[evaluator_id] = {
                        "score": 0.0,
                        "error": str(e)
                    }
            
            # Add example to results
            results["models"][model_id]["examples"].append({
                "input": input_text,
                "output": model_output,
                "reference_output": reference_output,
                "scores": example_scores
            })
        
        # Calculate statistics for each evaluator
        for evaluator_id, scores in all_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                
                # Calculate standard deviation
                variance = sum((x - avg_score) ** 2 for x in scores) / len(scores)
                std_dev = variance ** 0.5
                
                results["models"][model_id]["scores"][evaluator_id] = {
                    "average": round(avg_score, 4),
                    "min": round(min_score, 4),
                    "max": round(max_score, 4),
                    "std": round(std_dev, 4)
                }
                
                # Update overall evaluator scores
                results["evaluator_scores"][evaluator_id][model_id] = round(avg_score, 4)
    
    # Generate summary
    summary = generate_summary(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{dataset_type}_eval_{timestamp}.json"
    result_path = os.path.join(RESULTS_DIR, result_filename)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_filename = f"{dataset_type}_eval_{timestamp}_summary.json"
    summary_path = os.path.join(RESULTS_DIR, summary_filename)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation complete. Results saved to {result_path}")
    logger.info(f"Summary saved to {summary_path}")
    
    return True

def generate_summary(results):
    """Generate a summary of evaluation results."""
    summary = {
        "run_name": results["run_name"],
        "timestamp": results["timestamp"],
        "dataset_type": results["dataset_type"],
        "dataset_size": results["dataset_size"],
        "models_compared": list(results["models"].keys()),
        "metrics": {}
    }
    
    # Calculate metrics for each evaluator
    for evaluator_id, scores in results["evaluator_scores"].items():
        summary["metrics"][evaluator_id] = {}
        
        # Copy scores for each model
        for model_id, score in scores.items():
            summary["metrics"][evaluator_id][model_id] = score
        
        # If we have both teacher and student models, calculate differences
        if "teacher" in scores and "student" in scores:
            teacher_score = scores["teacher"]
            student_score = scores["student"]
            
            difference = teacher_score - student_score
            percent_difference = (difference / teacher_score) * 100 if teacher_score > 0 else 0
            
            summary["metrics"][evaluator_id]["difference"] = round(difference, 4)
            summary["metrics"][evaluator_id]["percent_difference"] = round(percent_difference, 2)
    
    # Generate conclusion
    conclusion = "Evaluation completed successfully. "
    
    if "teacher" in results["models"] and "student" in results["models"]:
        teacher_name = results["models"]["teacher"].get("name", "Teacher model")
        student_name = results["models"]["student"].get("name", "Student model")
        
        # Compare performance
        better_count = 0
        worse_count = 0
        total_metrics = 0
        
        for evaluator_id, metrics in summary["metrics"].items():
            if "difference" in metrics:
                total_metrics += 1
                if metrics["difference"] > 0:
                    better_count += 1
                else:
                    worse_count += 1
        
        if better_count > worse_count:
            conclusion += f"The {teacher_name} outperforms the {student_name} on {better_count}/{total_metrics} metrics. "
            
            # Add specific details about the biggest difference
            max_diff_metric = max(
                summary["metrics"].items(), 
                key=lambda x: x[1].get("percent_difference", 0) if "percent_difference" in x[1] else 0
            )[0]
            
            max_diff = summary["metrics"][max_diff_metric].get("percent_difference", 0)
            conclusion += f"The largest performance gap is in the {max_diff_metric} metric, where the teacher model scores {max_diff:.1f}% higher."
        else:
            conclusion += f"The {student_name} performs comparably to or better than the {teacher_name} on {total_metrics - better_count}/{total_metrics} metrics. "
            
            if better_count > 0:
                # Add specific details about where teacher is still better
                max_diff_metric = max(
                    summary["metrics"].items(), 
                    key=lambda x: x[1].get("percent_difference", 0) if "percent_difference" in x[1] else 0
                )[0]
                
                max_diff = summary["metrics"][max_diff_metric].get("percent_difference", 0)
                conclusion += f"The teacher model still performs better on the {max_diff_metric} metric by {max_diff:.1f}%."
    
    summary["conclusion"] = conclusion
    
    # Generate improvement suggestions
    if "student" in results["models"]:
        if results["dataset_type"] == "error_suggestion":
            summary["improvement_suggestions"] = (
                "The Student model could be improved by:\n"
                "1. Adding more specific troubleshooting commands in its responses\n"
                "2. Providing more comprehensive explanations of error causes\n"
                "3. Including follow-up steps for when initial solutions don't resolve the issue"
            )
        elif results["dataset_type"] == "command_extraction":
            summary["improvement_suggestions"] = (
                "The Student model could be improved by:\n"
                "1. Learning more precise command syntax and commonly used flags\n"
                "2. Prioritizing more robust command forms that handle edge cases better\n"
                "3. Focusing on modern command alternatives (like ss instead of netstat)"
            )
        else:
            summary["improvement_suggestions"] = (
                "The Student model could be improved by:\n"
                "1. Additional fine-tuning on domain-specific examples\n"
                "2. Improving response formatting to match expected outputs\n"
                "3. Enhancing reasoning capabilities for complex queries"
            )
    
    # Add chart data for visualization
    summary["charts"] = {
        "score_comparison": {
            "type": "bar",
            "data": {
                "labels": list(results["evaluator_scores"].keys()),
                "datasets": []
            }
        }
    }
    
    # Add datasets for each model
    for model_id, model_info in results["models"].items():
        dataset = {
            "label": model_info.get("name", model_id),
            "data": []
        }
        
        for evaluator_id in results["evaluator_scores"].keys():
            dataset["data"].append(results["evaluator_scores"][evaluator_id][model_id])
        
        summary["charts"]["score_comparison"]["data"]["datasets"].append(dataset)
    
    return summary

def create_mock_dataset(dataset_path, dataset_type):
    """Create a mock dataset for demonstration purposes."""
    examples = []
    
    if dataset_type == "error_suggestion":
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
            }
        ]
    elif dataset_type == "command_extraction":
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
            }
        ]
    else:
        examples = [
            {
                "input": "What is machine learning?",
                "reference": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."
            },
            {
                "input": "Explain how a neural network works",
                "reference": "A neural network is a computational model inspired by the human brain. It consists of layers of interconnected nodes (neurons) that process information and learn patterns from data."
            }
        ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # Write examples to JSONL file
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info(f"Created mock dataset with {len(examples)} examples at {dataset_path}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run OpenEvals evaluation")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--dataset", help="Dataset type to evaluate")
    parser.add_argument("--model", help="Model ID to evaluate")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error(f"Failed to load configuration from {args.config}")
        return 1
    
    # Run evaluation
    success = run_evaluation(config, args.dataset, args.model)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
