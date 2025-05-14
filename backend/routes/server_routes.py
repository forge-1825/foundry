from flask import Blueprint, jsonify
import docker
from ..config.server_config import SERVER_CONFIG, check_server_status

server_bp = Blueprint('server', __name__)
docker_client = docker.from_env()

@server_bp.route('/api/server-status', methods=['GET'])
def get_server_status():
    return jsonify(check_server_status())

@server_bp.route('/api/server/<server_type>/start', methods=['POST'])
def start_server(server_type):
    if server_type not in SERVER_CONFIG:
        return jsonify({'error': 'Invalid server type'}), 400
    
    try:
        container = docker_client.containers.get(SERVER_CONFIG[server_type]['container_name'])
        if container.status != 'running':
            container.start()
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@server_bp.route('/api/server/<server_type>/stop', methods=['POST'])
def stop_server(server_type):
    if server_type not in SERVER_CONFIG:
        return jsonify({'error': 'Invalid server type'}), 400
    
    try:
        container = docker_client.containers.get(SERVER_CONFIG[server_type]['container_name'])
        if container.status == 'running':
            container.stop()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
