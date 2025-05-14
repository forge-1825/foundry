from flask import Flask, jsonify, request, Blueprint
import os
import sys
import logging
import subprocess
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Root endpoint
@app.route('/')
def index():
    return jsonify({"message": "Model Distillation Pipeline API"})

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

# Script execution endpoint
@app.route('/api/scripts/<script_id>/execute', methods=['POST', 'OPTIONS'])
def execute_script(script_id):
    """Execute a script with the provided configuration."""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
        
    try:
        # Get the configuration from the request
        config = request.json
        logger.info(f"Executing script {script_id} with config: {config}")
        
        # Map of script IDs to actual script files
        SCRIPT_MAP = {
            'content_extraction_enrichment': 'run_data_enrichment.bat',
            'teacher_pair_generation': 'run_teacher_pair_generation_hierarchical.bat',
            'distillation': 'run_distillation_improved.bat',
            'merge_model': 'merge_model.py',
            'student_self_study': 'run_student_self_study_enhanced.bat',
            'evaluation': 'run_evaluation.py'
        }
        
        # Check if the script ID is valid
        if script_id not in SCRIPT_MAP:
            return jsonify({"error": f"Invalid script ID: {script_id}"}), 400
        
        # Handle content extraction and enrichment with our wrapper script
        if script_id == 'content_extraction_enrichment':
            # Extract parameters
            source_folder = config.get('source_folder', '')
            output_dir = config.get('output_dir', '')
            
            # Validate parameters
            if not source_folder:
                return jsonify({"error": "Source folder is required"}), 400
            
            # Set up the wrapper script path
            wrapper_script_path = os.path.join('scripts', 'run_data_enrichment_wrapper.py')
            
            # Check if the wrapper script exists
            if not os.path.exists(wrapper_script_path):
                return jsonify({"error": f"Wrapper script not found: {wrapper_script_path}"}), 404
            
            # Run the wrapper script
            try:
                # Build the command
                command = [sys.executable, wrapper_script_path, source_folder]
                if output_dir:
                    command.append(output_dir)
                
                logger.info(f"Running command: {command}")
                
                # Execute the command
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for the process to complete
                stdout, stderr = process.communicate()
                
                # Check if the process was successful
                if process.returncode != 0:
                    logger.error(f"Script execution failed: {stderr}")
                    return jsonify({
                        "error": f"Content extraction and enrichment failed",
                        "stdout": stdout,
                        "stderr": stderr
                    }), 500
                
                logger.info(f"Script execution successful: {stdout}")
                return jsonify({
                    "message": "Content extraction and enrichment completed successfully",
                    "stdout": stdout,
                    "stderr": stderr
                })
            
            except Exception as e:
                logger.error(f"Error running wrapper script: {e}")
                return jsonify({"error": f"Error running wrapper script: {str(e)}"}), 500
        
        # For other script types, use the original implementation
        # Get the script file path
        script_file = SCRIPT_MAP[script_id]
        
        # Try different possible locations for the script
        possible_paths = [
            os.path.join('..', script_file),  # Go up one level from the backend directory
            os.path.join('..', '..', script_file),  # Go up two levels
            os.path.join('..', 'scripts', script_file),  # Check in scripts directory
            script_file  # Check in current directory
        ]
        
        # Find the first path that exists
        script_path = None
        for path in possible_paths:
            if os.path.exists(path):
                script_path = path
                logger.info(f"Found script at: {script_path}")
                break
        
        # Check if the script file exists
        if not script_path or not os.path.exists(script_path):
            return jsonify({"error": f"Script file not found: {script_file}"}), 404
        
        # Prepare the command based on the script type
        if script_file.endswith('.bat'):
            # For batch files, we need to pass parameters as environment variables
            env = os.environ.copy()
            
            # Set environment variables based on the configuration
            # (This is for scripts other than content_extraction_enrichment)
            
            # Run the batch file with the environment variables
            process = subprocess.Popen(
                [script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True  # Required for batch files on Windows
            )
            
            # Wait for the process to complete
            stdout, stderr = process.communicate()
            
            # Check if the process was successful
            if process.returncode != 0:
                logger.error(f"Script execution failed: {stderr}")
                return jsonify({
                    "error": f"Script execution failed with return code {process.returncode}",
                    "stdout": stdout,
                    "stderr": stderr
                }), 500
            
            logger.info(f"Script execution successful: {stdout}")
            return jsonify({
                "message": "Script execution successful",
                "stdout": stdout,
                "stderr": stderr
            })
        
        elif script_file.endswith('.py'):
            # For Python files, we can execute them directly
            # TODO: Implement Python script execution
            return jsonify({"error": "Python script execution not implemented yet"}), 501
        
        else:
            return jsonify({"error": f"Unsupported script file type: {script_file}"}), 400
    
    except Exception as e:
        logger.error(f"Error executing script {script_id}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 7433))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True)
