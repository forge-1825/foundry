from flask import Blueprint, jsonify, request
import os
import json
import subprocess
import glob
import yaml
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

evaluation_bp = Blueprint('evaluation', __name__)

# Base directories
DATASETS_DIR = os.path.join(os.getcwd(), 'datasets')
RESULTS_DIR = os.path.join(os.getcwd(), 'results')
CONFIG_DIR = os.path.join(os.getcwd(), 'configs')

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Active evaluation processes
active_evaluations = {}

# In-memory storage for evaluation runs
evaluation_runs = {}

@evaluation_bp.route('/api/evaluation/datasets', methods=['GET'])
def get_datasets():
    """Get all available evaluation datasets."""
    try:
        datasets = []

        # Get all JSONL files in the datasets directory
        jsonl_files = glob.glob(os.path.join(DATASETS_DIR, '*.jsonl'))

        for file_path in jsonl_files:
            file_name = os.path.basename(file_path)
            dataset_type = "unknown"
            category = "standard"

            # Determine dataset type from filename
            if "simple_qa" in file_name:
                dataset_type = "simple_qa"
            elif "error_suggestion" in file_name:
                dataset_type = "error_suggestion"
            elif "command_extraction" in file_name:
                dataset_type = "command_extraction"
            elif "rag_" in file_name:
                dataset_type = "rag"
                category = "rag"

                # Determine RAG test type from filename
                if "standard" in file_name:
                    test_type = "standard"
                elif "noisy" in file_name:
                    test_type = "noisy_retrieval"
                elif "contradictory" in file_name:
                    test_type = "contradictory_information"
                elif "no_info" in file_name:
                    test_type = "information_not_present"
                elif "precision" in file_name:
                    test_type = "precision_test"
                else:
                    test_type = "standard"

            # Count examples in the dataset
            example_count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            example_count += 1
            except Exception as e:
                logger.error(f"Error counting examples in {file_name}: {e}")

            # Create dataset object
            dataset = {
                "id": file_name.replace('.jsonl', ''),
                "name": file_name,
                "type": dataset_type,
                "path": file_path,
                "example_count": example_count,
                "category": category
            }

            # Add test_type for RAG datasets
            if dataset_type == "rag":
                dataset["test_type"] = test_type

            datasets.append(dataset)

        # Add default RAG datasets if no RAG datasets were found
        if not any(d["type"] == "rag" for d in datasets):
            # Add default RAG datasets
            rag_datasets = [
                {
                    "id": "rag_standard",
                    "name": "RAG Standard Queries",
                    "type": "rag",
                    "category": "rag",
                    "test_type": "standard",
                    "example_count": 20
                },
                {
                    "id": "rag_noisy",
                    "name": "RAG Noisy Retrieval",
                    "type": "rag",
                    "category": "rag",
                    "test_type": "noisy_retrieval",
                    "example_count": 15
                },
                {
                    "id": "rag_contradictory",
                    "name": "RAG Contradictory Info",
                    "type": "rag",
                    "category": "rag",
                    "test_type": "contradictory_information",
                    "example_count": 15
                },
                {
                    "id": "rag_no_info",
                    "name": "RAG Missing Information",
                    "type": "rag",
                    "category": "rag",
                    "test_type": "information_not_present",
                    "example_count": 10
                },
                {
                    "id": "rag_precision",
                    "name": "RAG Precision Test",
                    "type": "rag",
                    "category": "rag",
                    "test_type": "precision_test",
                    "example_count": 10
                }
            ]
            datasets.extend(rag_datasets)

        return jsonify(datasets)
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/results', methods=['GET'])
def get_evaluation_results():
    """Get all available evaluation results."""
    try:
        results = []

        # Get all JSON files in the results directory
        json_files = glob.glob(os.path.join(RESULTS_DIR, '*.json'))

        for file_path in json_files:
            file_name = os.path.basename(file_path)

            # Skip summary files, we'll include them with their parent
            if file_name.endswith('_summary.json'):
                continue

            # Try to load the JSON to get metadata
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Check if there's a corresponding summary file
                summary_path = file_path.replace('.json', '_summary.json')
                has_summary = os.path.exists(summary_path)

                # Get timestamp from file or from content
                timestamp = data.get('timestamp',
                                    datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat())

                # Extract key information
                results.append({
                    "id": file_name.replace('.json', ''),
                    "name": data.get('run_name', file_name),
                    "timestamp": timestamp,
                    "dataset_size": data.get('dataset_size', 0),
                    "models": list(data.get('models', {}).keys()),
                    "dataset_type": data.get('dataset_type', 'unknown'),
                    "path": file_path,
                    "has_summary": has_summary,
                    "summary_path": summary_path if has_summary else None
                })
            except Exception as e:
                logger.error(f"Error parsing result file {file_name}: {e}")
                # Include basic info even if parsing failed
                results.append({
                    "id": file_name.replace('.json', ''),
                    "name": file_name,
                    "timestamp": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                    "path": file_path,
                    "error": str(e)
                })

        # Sort by timestamp, newest first
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return jsonify(results)
    except Exception as e:
        logger.error(f"Error getting evaluation results: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/result/<result_id>', methods=['GET'])
def get_evaluation_result(result_id):
    """Get a specific evaluation result."""
    try:
        # Construct the file path
        file_path = os.path.join(RESULTS_DIR, f"{result_id}.json")

        if not os.path.exists(file_path):
            return jsonify({"error": f"Result {result_id} not found"}), 404

        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if there's a summary file
        summary_path = file_path.replace('.json', '_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            data['summary'] = summary_data

        return jsonify(data)
    except Exception as e:
        logger.error(f"Error getting evaluation result {result_id}: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/models', methods=['GET'])
def get_evaluation_models():
    """Get all available models for evaluation."""
    try:
        # Query Docker for running containers
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}},{{.Ports}}"],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the output to find model containers
            models = []
            docker_output = result.stdout.strip().split('\n')

            for line in docker_output:
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) < 2:
                    continue

                container_name = parts[0]
                # ports = parts[1]  # Uncomment if needed in the future

                # Check for known model containers
                if "llama3_teacher" in container_name:
                    models.append({
                        "id": "teacher",
                        "name": "Llama 3 8B Instruct (Teacher)",
                        "endpoint": "http://localhost:8000/v1",
                        "model_id": "meta-llama/Llama-3-8B-Instruct",
                        "type": "teacher",
                        "container": container_name
                    })
                elif "phi3_vllm" in container_name and "distilled" not in container_name:
                    models.append({
                        "id": "student",
                        "name": "Phi-3 Mini (Student)",
                        "endpoint": "http://localhost:8002/v1",
                        "model_id": "microsoft/Phi-3-mini-4k-instruct",
                        "type": "student",
                        "container": container_name
                    })
                elif "phi3_distilled" in container_name:
                    models.append({
                        "id": "distilled",
                        "name": "Phi-3 Mini Distilled (TTRL)",
                        "endpoint": "http://localhost:8003/v1",
                        "model_id": "microsoft/Phi-3-mini-4k-instruct-distilled",
                        "type": "distilled",
                        "container": container_name
                    })

            # If no models were found, fall back to hardcoded list
            if not models:
                logger.warning("No model containers found. Using hardcoded list.")
                models = [
                    {
                        "id": "teacher",
                        "name": "Llama 3 8B Instruct (Teacher)",
                        "endpoint": "http://localhost:8000/v1",
                        "model_id": "meta-llama/Llama-3-8B-Instruct",
                        "type": "teacher"
                    },
                    {
                        "id": "student",
                        "name": "Phi-3 Mini (Student)",
                        "endpoint": "http://localhost:8002/v1",
                        "model_id": "microsoft/Phi-3-mini-4k-instruct",
                        "type": "student"
                    },
                    {
                        "id": "distilled",
                        "name": "Phi-3 Mini Distilled (TTRL)",
                        "endpoint": "http://localhost:8003/v1",
                        "model_id": "microsoft/Phi-3-mini-4k-instruct-distilled",
                        "type": "distilled"
                    }
                ]

        except Exception as docker_error:
            logger.error(f"Error querying Docker: {docker_error}")
            # Fall back to hardcoded list
            models = [
                {
                    "id": "teacher",
                    "name": "Llama 3 8B Instruct (Teacher)",
                    "endpoint": "http://localhost:8000/v1",
                    "model_id": "meta-llama/Llama-3-8B-Instruct",
                    "type": "teacher"
                },
                {
                    "id": "student",
                    "name": "Phi-3 Mini (Student)",
                    "endpoint": "http://localhost:8002/v1",
                    "model_id": "microsoft/Phi-3-mini-4k-instruct",
                    "type": "student"
                },
                {
                    "id": "distilled",
                    "name": "Phi-3 Mini Distilled (TTRL)",
                    "endpoint": "http://localhost:8003/v1",
                    "model_id": "microsoft/Phi-3-mini-4k-instruct-distilled",
                    "type": "distilled"
                }
            ]

        return jsonify(models)
    except Exception as e:
        logger.error(f"Error getting evaluation models: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/evaluators', methods=['GET'])
def get_evaluators():
    """Get all available evaluators."""
    try:
        # Return a list of available evaluators
        evaluators = [
            # Standard evaluators
            {
                "id": "string_match",
                "name": "String Match",
                "type": "string_distance",
                "description": "Simple string matching evaluator",
                "category": "standard"
            },
            {
                "id": "embedding_similarity",
                "name": "Embedding Similarity",
                "type": "embedding_distance",
                "description": "Semantic similarity evaluator using embeddings",
                "category": "standard"
            },
            {
                "id": "error_suggestion_judge",
                "name": "Error Suggestion Judge",
                "type": "llm_as_judge",
                "description": "LLM-as-judge evaluator for error suggestions",
                "category": "standard"
            },
            {
                "id": "command_extraction_judge",
                "name": "Command Extraction Judge",
                "type": "llm_as_judge",
                "description": "LLM-as-judge evaluator for command extraction",
                "category": "standard"
            },
            {
                "id": "qa_correctness",
                "name": "QA Correctness",
                "type": "llm_as_judge",
                "description": "LLM-as-judge evaluator for QA correctness",
                "category": "standard"
            },

            # RAG evaluators
            {
                "id": "retrieval_hit_rate",
                "name": "Retrieval Hit Rate",
                "type": "custom",
                "class": "RetrievalHitRateEvaluator",
                "description": "Evaluates if the retrieval system returns the expected documents",
                "category": "rag"
            },
            {
                "id": "retrieval_precision",
                "name": "Retrieval Precision",
                "type": "custom",
                "class": "RetrievalPrecisionEvaluator",
                "description": "Evaluates the precision of retrieved documents",
                "category": "rag"
            },
            {
                "id": "faithfulness",
                "name": "Faithfulness",
                "type": "custom",
                "class": "FaithfulnessEvaluator",
                "model": "teacher",
                "description": "Evaluates if the model's response is faithful to the retrieved context",
                "category": "rag"
            },
            {
                "id": "contradiction_handling",
                "name": "Contradiction Handling",
                "type": "custom",
                "class": "ContradictionHandlingEvaluator",
                "model": "teacher",
                "description": "Evaluates how well the model handles contradictory information",
                "category": "rag"
            },
            {
                "id": "noise_robustness",
                "name": "Noise Robustness",
                "type": "custom",
                "class": "NoiseRobustnessEvaluator",
                "model": "teacher",
                "description": "Evaluates how well the model handles noisy retrieval results",
                "category": "rag"
            },
            {
                "id": "no_info_handling",
                "name": "No Info Handling",
                "type": "custom",
                "class": "NoInfoHandlingEvaluator",
                "model": "teacher",
                "description": "Evaluates how well the model handles missing information",
                "category": "rag"
            }
        ]

        return jsonify(evaluators)
    except Exception as e:
        logger.error(f"Error getting evaluators: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/configs', methods=['GET'])
def get_evaluation_configs():
    """Get all available evaluation configurations."""
    try:
        configs = []

        # Get all YAML files in the config directory
        yaml_files = glob.glob(os.path.join(os.getcwd(), '*.yaml'))

        for file_path in yaml_files:
            file_name = os.path.basename(file_path)

            # Only include eval config files
            if file_name.startswith('eval_config'):
                configs.append({
                    "id": file_name.replace('.yaml', ''),
                    "name": file_name,
                    "path": file_path
                })

        return jsonify(configs)
    except Exception as e:
        logger.error(f"Error getting evaluation configs: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/run', methods=['POST'])
def run_evaluation():
    """Run an evaluation with the given configuration."""
    try:
        data = request.json

        # Check if this is a RAG evaluation
        is_rag_evaluation = data.get('is_rag_evaluation', False)

        if is_rag_evaluation:
            # RAG evaluation
            # Required parameters
            models = data.get('models', [])
            if not models:
                return jsonify({"error": "At least one model is required"}), 400

            # Optional parameters
            rag_topic = data.get('rag_topic', 'metasploit')
            rag_test_type = data.get('rag_test_type', 'all')
            rag_test_types = data.get('rag_test_types', [])

            # Prepare the command
            cmd = ["python", "run_rag_openevals.py", "--topic", rag_topic]

            # Add test type if specified
            if rag_test_type and rag_test_type != 'all':
                cmd.extend(["--test-type", rag_test_type])
            elif rag_test_types and len(rag_test_types) > 0:
                # If specific test types are provided from selected datasets, use them
                for test_type in rag_test_types:
                    cmd.extend(["--test-type", test_type])

            # Add models
            for model in models:
                cmd.extend(["--model", model['id']])

            # Add verbose flag
            cmd.append("--verbose")

            # Generate a unique ID for this evaluation
            evaluation_id = f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        else:
            # Standard evaluation
            # Required parameters
            config_path = data.get('config_path', 'eval_config_full_comparison.yaml')

            # Optional parameters
            dataset_type = data.get('dataset_type')
            model = data.get('model')

            # Prepare the command
            cmd = ["python", "run_openevals_langchain.py", "--config", config_path]

            if dataset_type:
                cmd.extend(["--dataset", dataset_type])

            if model:
                cmd.extend(["--model", model])

            # Generate a unique ID for this evaluation
            evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Start the process
        logger.info(f"Starting evaluation {evaluation_id} with command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Store the process
        active_evaluations[evaluation_id] = {
            "process": process,
            "command": ' '.join(cmd),
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "logs": []
        }

        # Start a thread to capture output
        def capture_output():
            for line in process.stdout:
                active_evaluations[evaluation_id]["logs"].append(line.strip())

            # Process completed
            return_code = process.wait()
            active_evaluations[evaluation_id]["status"] = "completed" if return_code == 0 else "error"
            active_evaluations[evaluation_id]["end_time"] = datetime.now().isoformat()
            active_evaluations[evaluation_id]["return_code"] = return_code

        import threading
        thread = threading.Thread(target=capture_output)
        thread.daemon = True
        thread.start()

        return jsonify({
            "evaluation_id": evaluation_id,
            "status": "running",
            "command": ' '.join(cmd)
        })
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/status/<evaluation_id>', methods=['GET'])
def get_evaluation_status(evaluation_id):
    """Get the status of a running evaluation."""
    try:
        if evaluation_id not in active_evaluations:
            return jsonify({"error": f"Evaluation {evaluation_id} not found"}), 404

        evaluation = active_evaluations[evaluation_id]

        # Check if the process is still running
        if evaluation["status"] == "running":
            # Check if the process is still alive
            if evaluation["process"].poll() is not None:
                # Process has completed
                return_code = evaluation["process"].poll()
                evaluation["status"] = "completed" if return_code == 0 else "error"
                evaluation["end_time"] = datetime.now().isoformat()
                evaluation["return_code"] = return_code

        # Return the status
        return jsonify({
            "evaluation_id": evaluation_id,
            "status": evaluation["status"],
            "start_time": evaluation["start_time"],
            "end_time": evaluation.get("end_time"),
            "return_code": evaluation.get("return_code"),
            "command": evaluation["command"],
            "log_count": len(evaluation["logs"])
        })
    except Exception as e:
        logger.error(f"Error getting evaluation status {evaluation_id}: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/logs/<evaluation_id>', methods=['GET'])
def get_evaluation_logs(evaluation_id):
    """Get the logs of a running evaluation."""
    try:
        if evaluation_id not in active_evaluations:
            return jsonify({"error": f"Evaluation {evaluation_id} not found"}), 404

        evaluation = active_evaluations[evaluation_id]

        # Get optional parameters
        start = request.args.get('start', 0, type=int)
        limit = request.args.get('limit', 100, type=int)

        # Return the logs
        logs = evaluation["logs"][start:start+limit]

        return jsonify({
            "evaluation_id": evaluation_id,
            "status": evaluation["status"],
            "logs": logs,
            "total_logs": len(evaluation["logs"]),
            "start": start,
            "limit": limit
        })
    except Exception as e:
        logger.error(f"Error getting evaluation logs {evaluation_id}: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/cancel/<evaluation_id>', methods=['POST'])
def cancel_evaluation(evaluation_id):
    """Cancel a running evaluation."""
    try:
        if evaluation_id not in active_evaluations:
            return jsonify({"error": f"Evaluation {evaluation_id} not found"}), 404

        evaluation = active_evaluations[evaluation_id]

        # Check if the process is still running
        if evaluation["status"] == "running":
            # Kill the process
            evaluation["process"].kill()
            evaluation["status"] = "cancelled"
            evaluation["end_time"] = datetime.now().isoformat()

        return jsonify({
            "evaluation_id": evaluation_id,
            "status": evaluation["status"]
        })
    except Exception as e:
        logger.error(f"Error cancelling evaluation {evaluation_id}: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/upload/dataset', methods=['POST'])
def upload_dataset():
    """Upload a new dataset file."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.endswith('.jsonl'):
            return jsonify({"error": "File must be a JSONL file"}), 400

        # Save the file
        file_path = os.path.join(DATASETS_DIR, file.filename)
        file.save(file_path)

        return jsonify({
            "message": f"File {file.filename} uploaded successfully",
            "path": file_path
        })
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/run', methods=['POST'])
def trigger_evaluation_run():
    """Trigger a new evaluation run."""
    try:
        data = request.json

        # Generate a unique run ID
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

        # Extract configuration parameters
        models = data.get('models', [])
        datasets = data.get('datasets', [])
        evaluators = data.get('evaluators', [])
        run_name = data.get('run_name', f"Evaluation Run {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Validate required parameters
        if not models:
            return jsonify({"error": "At least one model must be selected"}), 400
        if not datasets:
            return jsonify({"error": "At least one dataset must be selected"}), 400
        if not evaluators:
            return jsonify({"error": "At least one evaluator must be selected"}), 400

        # Create a temporary configuration file
        config = {
            "run_name": run_name,
            "description": f"Evaluation run triggered from UI at {datetime.now().isoformat()}",
            "models": {},
            "evaluators": evaluators,
            "max_examples": data.get('max_examples', 50)
        }

        # Add model configurations
        for model in models:
            model_id = model.get('id')
            if model_id:
                config["models"][model_id] = {
                    "name": model.get('name', model_id),
                    "endpoint": model.get('endpoint', "http://localhost:8000/v1"),
                    "model_id": model.get('model_id', model_id),
                    "type": model.get('type', "unknown")
                }

        # Create a temporary config file
        config_path = os.path.join(CONFIG_DIR, f"eval_config_{run_id}.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Store run information
        evaluation_runs[run_id] = {
            "id": run_id,
            "status": "queued",
            "start_time": datetime.now().isoformat(),
            "config": config,
            "datasets": datasets,
            "logs": [],
            "results": []
        }

        # Start a thread to run the evaluation
        def run_evaluation():
            try:
                # Update status to running
                evaluation_runs[run_id]["status"] = "running"

                # Run the evaluation for each dataset
                for dataset in datasets:
                    # Prepare the command
                    cmd = [
                        "python",
                        os.path.join(os.getcwd(), "scripts", "run_openevals_langchain.py"),
                        "--config", config_path,
                        "--dataset", dataset
                    ]

                    # Log the command
                    log_entry = f"[{datetime.now().isoformat()}] Running command: {' '.join(cmd)}"
                    evaluation_runs[run_id]["logs"].append(log_entry)
                    logger.info(log_entry)

                    # Run the command
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )

                    # Capture output
                    for line in process.stdout:
                        log_entry = f"[{datetime.now().isoformat()}] {line.strip()}"
                        evaluation_runs[run_id]["logs"].append(log_entry)

                    # Wait for the process to complete
                    return_code = process.wait()

                    # Check if successful
                    if return_code != 0:
                        log_entry = f"[{datetime.now().isoformat()}] Evaluation failed with return code {return_code}"
                        evaluation_runs[run_id]["logs"].append(log_entry)
                        logger.error(log_entry)
                        evaluation_runs[run_id]["status"] = "failed"
                        return

                    # Look for result files
                    result_files = glob.glob(os.path.join(RESULTS_DIR, f"{dataset}_eval_*.json"))
                    result_files.sort(key=os.path.getmtime, reverse=True)

                    if result_files:
                        # Get the most recent result file
                        latest_result = result_files[0]
                        evaluation_runs[run_id]["results"].append({
                            "dataset": dataset,
                            "result_file": os.path.basename(latest_result)
                        })

                        log_entry = f"[{datetime.now().isoformat()}] Evaluation completed successfully for dataset {dataset}. Results saved to {latest_result}"
                        evaluation_runs[run_id]["logs"].append(log_entry)
                        logger.info(log_entry)
                    else:
                        log_entry = f"[{datetime.now().isoformat()}] Evaluation completed but no result file found for dataset {dataset}"
                        evaluation_runs[run_id]["logs"].append(log_entry)
                        logger.warning(log_entry)

                # Update status to completed
                evaluation_runs[run_id]["status"] = "completed"
                evaluation_runs[run_id]["end_time"] = datetime.now().isoformat()

                # Log completion
                log_entry = f"[{datetime.now().isoformat()}] All evaluations completed successfully"
                evaluation_runs[run_id]["logs"].append(log_entry)
                logger.info(log_entry)

            except Exception as e:
                # Log error
                log_entry = f"[{datetime.now().isoformat()}] Error running evaluation: {str(e)}"
                evaluation_runs[run_id]["logs"].append(log_entry)
                logger.error(log_entry)

                # Update status to failed
                evaluation_runs[run_id]["status"] = "failed"
                evaluation_runs[run_id]["end_time"] = datetime.now().isoformat()
                evaluation_runs[run_id]["error"] = str(e)

        # Start the evaluation in a separate thread
        import threading
        thread = threading.Thread(target=run_evaluation)
        thread.daemon = True
        thread.start()

        return jsonify({
            "run_id": run_id,
            "status": "queued",
            "message": f"Evaluation run {run_id} queued successfully"
        })
    except Exception as e:
        logger.error(f"Error triggering evaluation run: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/run/<run_id>/status', methods=['GET'])
def get_evaluation_run_status(run_id):
    """Get the status of an evaluation run."""
    try:
        if run_id not in evaluation_runs:
            return jsonify({"error": f"Evaluation run {run_id} not found"}), 404

        run = evaluation_runs[run_id]

        response = {
            "run_id": run_id,
            "status": run["status"],
            "start_time": run["start_time"],
            "config": run["config"],
            "datasets": run["datasets"],
            "log_count": len(run["logs"]),
            "results": run["results"]
        }

        if "end_time" in run:
            response["end_time"] = run["end_time"]

        if "error" in run:
            response["error"] = run["error"]

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting evaluation run status: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/run/<run_id>/logs', methods=['GET'])
def get_evaluation_run_logs(run_id):
    """Get the logs of an evaluation run."""
    try:
        if run_id not in evaluation_runs:
            return jsonify({"error": f"Evaluation run {run_id} not found"}), 404

        run = evaluation_runs[run_id]

        # Get optional parameters
        start = request.args.get('start', 0, type=int)
        limit = request.args.get('limit', 100, type=int)

        # Return the logs
        logs = run["logs"][start:start+limit]

        return jsonify({
            "run_id": run_id,
            "status": run["status"],
            "logs": logs,
            "total_logs": len(run["logs"]),
            "start": start,
            "limit": limit
        })
    except Exception as e:
        logger.error(f"Error getting evaluation run logs: {e}")
        return jsonify({"error": str(e)}), 500

@evaluation_bp.route('/api/evaluation/run/<run_id>/cancel', methods=['POST'])
def cancel_evaluation_run(run_id):
    """Cancel an evaluation run."""
    try:
        if run_id not in evaluation_runs:
            return jsonify({"error": f"Evaluation run {run_id} not found"}), 404

        run = evaluation_runs[run_id]

        # Check if the run is still in progress
        if run["status"] in ["queued", "running"]:
            # Update status to cancelled
            run["status"] = "cancelled"
            run["end_time"] = datetime.now().isoformat()

            # Log cancellation
            log_entry = f"[{datetime.now().isoformat()}] Evaluation run cancelled by user"
            run["logs"].append(log_entry)
            logger.info(log_entry)

            return jsonify({
                "run_id": run_id,
                "status": "cancelled",
                "message": f"Evaluation run {run_id} cancelled successfully"
            })
        else:
            return jsonify({
                "run_id": run_id,
                "status": run["status"],
                "message": f"Evaluation run {run_id} is already {run['status']}"
            })
    except Exception as e:
        logger.error(f"Error cancelling evaluation run: {e}")
        return jsonify({"error": str(e)}), 500
