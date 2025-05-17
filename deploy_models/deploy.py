import paramiko
import re
import time
import os
from typing import Dict, Optional,List
from google.cloud import firestore
from db.firebase import db
import logging
from datetime import datetime
import requests
from utility.utils import extract_ssh_from_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_model(node: dict, model: dict, is_gpu: bool, ngrok_token: str) -> str:
    """
    Deploy a model to a remote node via SSH, installing Docker if needed, and running the container.

    Args:
        node: Dict containing node details with 'node_id' and 'ssh' (format: ssh://username@ip:port).
        model: Dict containing 'name' and 'model_path'.
        is_gpu: Bool indicating if the node uses GPU resources.
        ngrok_token: NGROK authentication token.

    Returns:
        str: Newline-separated status messages indicating success or failure for the node.
    """
    logger = logging.getLogger(__name__)
    results = []
    node_id = node.get("node_id")
    ssh_val = node['compute_resources'][0]['network']['ssh']

    if not node_id:
        return f"Failed to deploy {model['name']}: Missing node_id in node info"
    if not ssh_val:
        return f"Failed to deploy {model['name']}: No SSH info provided"
    if not isinstance(ssh_val, str):
        return f"Failed to deploy {model['name']} to node {node_id}: SSH info must be a string, got {type(ssh_val)}"

    # Parse SSH string (format: ssh://username@ip:port)
    ssh_pattern = r'^ssh://([^@]+)@([^:]+):(\d+)$'
    match = re.match(ssh_pattern, ssh_val)
    if not match:
        return f"Failed to deploy {model['name']} to node {node_id}: Invalid SSH string format: {ssh_val}"

    username, ip_address, port = match.group(1), match.group(2), int(match.group(3))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    container_id = None
    try:
        logger.info(f"Connecting to node {node_id} at {ip_address}:{port} as {username}")
        client.connect(hostname=ip_address, port=port, username=username, key_filename="deploy_models/ssh_host_key", timeout=10)

        # Check resources
        logger.info(f"Checking resources on node {node_id}")
        stdin, stdout, stderr = client.exec_command("free -m | grep Mem | awk '{print $4}'")
        available_memory_mb = int(stdout.read().decode().strip() or 0)
        stdin, stdout, stderr = client.exec_command("df -h / | tail -n 1 | awk '{print $4}' | sed 's/G//'")
        available_disk_gb = float(stdout.read().decode().strip() or 0)
        if available_memory_mb < 2048 or available_disk_gb < 10:
            return f"Failed to deploy {model['name']} to node {node_id}: Insufficient resources (Memory: {available_memory_mb}MB, Disk: {available_disk_gb}GB)"

        # Check and install Docker
        stdin, stdout, stderr = client.exec_command("docker --version")
        if "Docker version" not in stdout.read().decode():
            logger.info(f"Installing Docker on node {node_id}")
            stdin, stdout, stderr = client.exec_command("curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh")
            if stderr.read().decode().strip():
                return f"Failed to deploy {model['name']} to node {node_id}: Docker installation failed"
            time.sleep(15)

            # Ensure user has Docker permissions
            logger.info(f"Ensuring {username} has Docker permissions on node {node_id}")
            stdin, stdout, stderr = client.exec_command(f"getent group docker | grep {username}")
            if not stdout.read().decode().strip():
                logger.info(f"Adding {username} to docker group on node {node_id}")
                stdin, stdout, stderr = client.exec_command(f"sudo usermod -aG docker {username}")
                if stderr.read().decode().strip():
                    return f"Failed to deploy {model['name']} to node {node_id}: Failed to add user to docker group"
                
                # Reconnect to apply group changes
                client.close()
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(hostname=ip_address, port=port, username=username, key_filename="deploy_models/ssh_host_key", timeout=10)

        # Start Docker service
        client.exec_command("sudo systemctl start docker || true")
        client.exec_command("sudo systemctl enable docker || true")
        time.sleep(3)

        # Verify Docker daemon
        stdin, stdout, stderr = client.exec_command("docker info --format '{{.ServerVersion}}'")
        if not stdout.read().decode().strip():
            return f"Failed to deploy {model['name']} to node {node_id}: Docker daemon not running"

        # Pull Docker image
        logger.info(f"Verifying bateesa/polaris-inference image on node {node_id}")
        stdin, stdout, stderr = client.exec_command("docker images -q bateesa/polaris-inference")
        image_id = stdout.read().decode().strip()
        if not image_id:
            logger.info(f"Pulling bateesa/polaris-inference image on node {node_id}")
            stdin, stdout, stderr = client.exec_command("docker pull bateesa/polaris-inference")
            start_time = time.time()
            while time.time() - start_time < 300:
                stdin, stdout, stderr = client.exec_command("docker images -q bateesa/polaris-inference")
                image_id = stdout.read().decode().strip()
                if image_id:
                    break
                time.sleep(5)
            else:
                return f"Failed to deploy {model['name']} to node {node_id}: Image pull timeout"

        # Find available port
        initial_port = 8094
        port = initial_port
        while True:
            stdin, stdout, stderr = client.exec_command(f"ss -tuln | grep :{port} || netstat -tuln | grep :{port}")
            if not stdout.read().decode().strip():
                break
            port += 1
            if port > initial_port + 100:
                return f"Failed to deploy {model['name']} to node {node_id}: No available ports found"

        # Run Docker container
        model_id = model["model_path"].replace("/", "_")
        gpu_flag = "--gpus all" if is_gpu else ""
        command = (
            f"docker run -d -i -t -p {port}:{port} "
            f"-e PORT={port} "
            f"-e NGROK_AUTH_TOKEN={ngrok_token} "
            f"-e MODEL_NAME={model['model_path']} "
            f"-e MODEL_ID={model_id} "
            f"{gpu_flag} bateesa/polaris-inference"
        )

        logger.info(f"Executing Docker command on node {node_id}: {command}")
        stdin, stdout, stderr = client.exec_command(command)
        container_id = stdout.read().decode().strip()
        if not container_id:
            return f"Failed to deploy {model['name']} to node {node_id}: Could not start container"

        # Verify container status
        stdin, stdout, stderr = client.exec_command(f"docker inspect --format='{{{{.State.Status}}}}' {container_id}")
        container_status = stdout.read().decode().strip()
        if container_status != "running":
            stdin, stdout, stderr = client.exec_command(f"docker logs {container_id}")
            container_logs = stdout.read().decode() + stderr.read().decode()
            return f"Failed to deploy {model['name']} to node {node_id}: Container {container_id} is {container_status}: {container_logs}"

        return f"Successfully deployed {model['name']} to node {node_id} on port {port} (Container ID: {container_id})"

    except paramiko.AuthenticationException as e:
        return f"Failed to deploy {model['name']} to node {node_id}: SSH authentication failed: {str(e)}"
    except paramiko.SSHException as e:
        return f"Failed to deploy {model['name']} to node {node_id}: SSH connection failed: {str(e)}"
    except KeyboardInterrupt:
        logger.info(f"Deployment interrupted by user for node {node_id}. Cleaning up...")
        if container_id:
            client.exec_command(f"docker rm -f {container_id}")
        return f"Failed to deploy {model['name']} to node {node_id}: Deployment interrupted"
    except Exception as e:
        return f"Failed to deploy {model['name']} to node {node_id}: {str(e)}"
    finally:
        client.close()

def update_assignment_in_firestore(node_id: str, model_name: str, model_type: str, hot: bool, db: firestore.Client):
    """
    Update or create an assignment document in Firestore for a node.
    
    Args:
        node_id: The ID of the node.
        model_name: The name of the model to assign.
        model_type: Either "gpu" or "cpu".
        hot: Whether the model is hot.
        db: Firestore client instance.
    """
    try:
        assignment_ref = db.collection("assignments").document(node_id)
        assignment_doc = assignment_ref.get()
        
        assignment_data = {
            "node_id": node_id,
            "type": model_type.lower(),
            "models": [],
            "timestamp": datetime.utcnow()
        }
        
        if assignment_doc.exists:
            existing_data = assignment_doc.to_dict()
            assignment_data["models"] = existing_data.get("models", [])
            assignment_data["type"] = existing_data.get("type", model_type.lower())
            
            # Update existing model or append new one
            for model in assignment_data["models"]:
                if model["model"] == model_name:
                    model["hot"] = hot
                    break
            else:
                assignment_data["models"].append({"model": model_name, "hot": hot})
        else:
            assignment_data["models"].append({"model": model_name, "hot": hot})
        
        assignment_ref.set(assignment_data, merge=True)
        logger.info(f"Updated assignment for node {node_id}: {assignment_data}")
    except Exception as e:
        logger.error(f"Failed to update assignment for node {node_id}: {str(e)}")


def update_node_and_assignment(node_id: str, model_name: str, model_type: str, hot: bool, db: firestore.Client):
    try:
        node_ref = db.collection("nodes").document(node_id)
        node_ref.update({"hot": True})
        logger.info(f"Updated hot status to True for node {node_id}")
        
        assignment_ref = db.collection("assignments").document(node_id)
        assignment_doc = assignment_ref.get()
        assignment_data = {
            "node_id": node_id,
            "type": model_type.lower(),
            "models": [],
            "timestamp": datetime.utcnow()
        }
        
        if assignment_doc.exists:
            existing_data = assignment_doc.to_dict()
            assignment_data["models"] = existing_data.get("models", [])
            assignment_data["type"] = existing_data.get("type", model_type.lower())
            for model in assignment_data["models"]:
                if model["model"] == model_name:
                    model["hot"] = hot
                    break
            else:
                assignment_data["models"].append({"model": model_name, "hot": hot})
        else:
            assignment_data["models"].append({"model": model_name, "hot": hot})
        
        assignment_ref.set(assignment_data, merge=True)
        logger.info(f"Updated assignment for node {node_id}: {assignment_data}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to update node {node_id} hot status or assignment: {str(e)}")
        return False