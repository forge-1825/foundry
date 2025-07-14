from flask import Flask, jsonify, request, send_from_directory
import os
import logging
import subprocess
import json
from routes.evaluation_routes import evaluation_bp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Register blueprints
app.register_blueprint(evaluation_bp)

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

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 7433))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=True)
