import paramiko
from typing import List, Dict, Optional,Tuple
from dataclasses import dataclass
from google.cloud import firestore
import re
import time
from db.firebase import db
import logging
from typing import Dict, List
from models import *


# Data classes for assignments
@dataclass
class ModelAssignment:
    model: str
    hot: bool

@dataclass
class NodeAssignment:
    node_id: str
    models: List[ModelAssignment]

# Constants
MIN_REDUNDANCY = 2
UTILIZATION_THRESHOLD_LOW = 0.2  # 20% for scale-down
UTILIZATION_THRESHOLD_HIGH = 0.8  # 80% for scale-up
HIGH_DEMAND_MODELS = set()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("deploy_model.log")
    ]
)
logger = logging.getLogger(__name__)

def load_data_from_db(db: firestore.Client) -> Tuple[Dict, Dict, List[str], List[Dict]]:
    """
    Load GPU models, CPU models, node IDs, and node data from Firestore.
    
    Args:
        db: Firestore client instance.
    
    Returns:
        Tuple[Dict, Dict, List[str], List[Dict]]:
            - gpu_models: Dictionary of GPU models categorized by type.
            - cpu_models: Dictionary of CPU models categorized by type and parameters.
            - node_ids: List of node IDs from the nodes collection.
            - nodes_list: List of dictionaries containing full node data from the nodes collection.
    """
    logger.debug("Starting to load data from Firestore")

    # Load GPU models
    gpu_models = {
        "multimodal_models": {
            "qwen_models": [],
            "deepseek_models": [],
            "llama_models": [],
            "internvl_models": [],
            "phi_models": [],
            "other_models": []
        }
    }
    try:
        for doc in db.collection("gpu_models").stream():
            model = doc.to_dict()
            gpu_models["multimodal_models"]["other_models"].append(model)
        logger.info(f"Loaded {len(gpu_models['multimodal_models']['other_models'])} GPU models")
    except Exception as e:
        logger.error(f"Error loading GPU models: {str(e)}")
        gpu_models["multimodal_models"]["other_models"] = []

    # Load CPU models
    cpu_models = {
        "text_models": {
            "llama_models": [],
            "mistral_models": [],
            "phi_models": [],
            "gemma_models": [],
            "qwen_models": [],
            "yi_models": [],
            "other_models": []
        },
        "specialized_models": {
            "coding_models": [],
            "math_models": [],
            "multilingual_models": []
        },
        "lightweight_models": {
            "tiny_models": [],
            "context_optimized": []
        }
    }
    try:
        for doc in db.collection("cpu_models").stream():
            model = doc.to_dict()
            model_type = model.get("type", "")
            parameters = model.get("parameters", "")
            if model_type in ["coding", "math", "multilingual"]:
                cpu_models["specialized_models"][f"{model_type}_models"].append(model)
            elif parameters in ["1.1B", "1B", "3B"] and model_type in ["chat", "instruct"]:
                cpu_models["lightweight_models"]["tiny_models"].append(model)
            else:
                cpu_models["text_models"]["other_models"].append(model)
        logger.info(
            f"Loaded CPU models: "
            f"{len(cpu_models['text_models']['other_models'])} text, "
            f"{len(cpu_models['specialized_models']['coding_models'])} coding, "
            f"{len(cpu_models['lightweight_models']['tiny_models'])} tiny"
        )
    except Exception as e:
        logger.error(f"Error loading CPU models: {str(e)}")
        cpu_models["text_models"]["other_models"] = []

    # Load nodes
    node_ids = []
    nodes_list = []
    try:
        for doc in db.collection("nodes").stream():
            node_data = doc.to_dict()
            node_id = doc.id
            # Ensure node_id is included in the dictionary
            node_data["node_id"] = node_id
            node_ids.append(node_id)
            nodes_list.append(node_data)
        logger.info(f"Loaded {len(node_ids)} node IDs and {len(nodes_list)} nodes from Firestore")
    except Exception as e:
        logger.error(f"Error loading nodes: {str(e)}")
        node_ids = []
        nodes_list = []

    # Load high-demand models
    global HIGH_DEMAND_MODELS
    try:
        doc = db.collection("high_demand_models").document("config").get()
        if doc.exists:
            HIGH_DEMAND_MODELS = set(doc.to_dict().get("models", []))
            logger.info(f"Loaded {len(HIGH_DEMAND_MODELS)} high-demand models")
        else:
            # Initialize with empty list if not exists
            db.collection("high_demand_models").document("config").set({"models": []})
            HIGH_DEMAND_MODELS = set()
            logger.info("Initialized empty high-demand models config")
    except Exception as e:
        logger.error(f"Error loading high-demand models: {str(e)}")
        HIGH_DEMAND_MODELS = set()

    logger.debug("Completed loading data from Firestore")
    return gpu_models, cpu_models, node_ids, nodes_list

def update_high_demand_models(db: firestore.Client, models: List[str]):
    """
    Update the in-memory HIGH_DEMAND_MODELS and Firestore.
    """
    global HIGH_DEMAND_MODELS
    HIGH_DEMAND_MODELS = set(models)
    db.collection("high_demand_models").document("config").set({"models": list(models)})
def estimate_model_resources(model: Dict, is_gpu: bool) -> Dict:
    params_str = model["parameters"].replace("B", "").replace(" MoE", "")
    try:
        params = float(params_str)
    except ValueError:
        params = 7.0
    if is_gpu:
        vram = params * 2 * 1.2
        storage = vram * 0.5
        ram = params * 0.1
    else:
        ram = params * 0.7 * 1.3
        storage = ram * 0.8
        vram = 0
    return {"vram": vram, "ram": ram, "storage": storage}

def get_node_utilization(node_id: str, resource_type: str) -> Dict:
    """
    Fetch node utilization based on the number of models assigned in the Firestore 'assignments' collection.
    Excludes nodes with more than 4 models by marking them as fully utilized.
    
    Args:
        node_id: str, document ID of the node from the 'nodes' collection.
        resource_type: str, either 'CPU' or 'GPU'.
        db: Firestore client instance.
    
    Returns:
        Dict with 'cpu_utilization' and 'gpu_utilization' as floats (0.0 to 1.0).
    """
    try:
        # Fetch assignment document for the node
        assignment_doc = db.collection("assignments").document(node_id).get()
        
        # Initialize metrics
        metrics = {"cpu_utilization": 0.0, "gpu_utilization": 0.0}
        
        if assignment_doc.exists:
            assignment_data = assignment_doc.to_dict()
            models = assignment_data.get("models", [])
            model_count = len(models)
            
            # If more than 4 models are assigned, mark node as fully utilized
            if model_count > 4:
                return {
                    "cpu_utilization": 1.0 if resource_type == "CPU" else 0.0,
                    "gpu_utilization": 1.0 if resource_type == "GPU" else 0.0
                }
            print(f"deployment counts {model_count}")
            # Calculate utilization as a fraction of max models (e.g., 4 models = max capacity)
            utilization = model_count / 4.0  # Linear scaling: 0 models = 0.0, 4 models = 1.0
            if resource_type == "CPU":
                metrics["cpu_utilization"] = utilization
            else:
                metrics["gpu_utilization"] = utilization
        
        return metrics
    
    except Exception as e:
        print(f"Error fetching utilization for node {node_id}: {str(e)}")
        return {"cpu_utilization": 0.0, "gpu_utilization": 0.0}

def get_available_nodes(nodes_list, resource_type: str ) -> list:
    """
    Get a list of available nodes based on their status and resource type from Firestore.
    
    Args:
        node_ids: List[str], list of node document IDs from 'nodes' collection.
        resource_type: str, either 'CPU' or 'GPU'.
        usage_data: List[Dict], usage data for sorting (not used here).
        db: Firestore client instance.
    
    Returns:
        List[Dict], list of node dictionaries that are available, with node_id attached.
    """
    available = []
    node_metrics = {}
    for node_data in nodes_list:
        node_id =node_data["node_id"]
        if (node_data.get("status") == "online" and node_data.get("compute_resources", [{}])[0].get("resource_type") == resource_type):
            available.append(node_data)
            metrics = get_node_utilization(node_id, resource_type)
            node_metrics[node_id] = metrics

    # Sort available nodes by utilization, then by memory or uptime
    return sorted(
        available,
        key=lambda x: (
            node_metrics[x["node_id"]]["gpu_utilization" if resource_type == "GPU" else "cpu_utilization"],
            float(x["compute_resources"][0]["gpu_specs"]["memory_size"].replace("GB", "")) if resource_type == "GPU"
            else float(x["compute_resources"][0]["ram"].replace("GB", "")),
            x.get("uptime_seconds", 0)
        )
    )


def flatten_models(gpu_models: Dict, cpu_models: Dict) -> tuple[List[Dict], List[Dict]]:
    gpu_model_list = (
        gpu_models["multimodal_models"]["qwen_models"] +
        gpu_models["multimodal_models"]["deepseek_models"] +
        gpu_models["multimodal_models"]["llama_models"] +
        gpu_models["multimodal_models"]["internvl_models"] +
        gpu_models["multimodal_models"]["phi_models"] +
        gpu_models["multimodal_models"]["other_models"]
    )
    cpu_model_list = (
        cpu_models["text_models"]["llama_models"] +
        cpu_models["text_models"]["mistral_models"] +
        cpu_models["text_models"]["phi_models"] +
        cpu_models["text_models"]["gemma_models"] +
        cpu_models["text_models"]["qwen_models"] +
        cpu_models["text_models"]["yi_models"] +
        cpu_models["specialized_models"]["coding_models"] +
        cpu_models["specialized_models"]["math_models"] +
        cpu_models["specialized_models"]["multilingual_models"] +
        cpu_models["lightweight_models"]["tiny_models"] +
        cpu_models["lightweight_models"]["context_optimized"] +
        cpu_models["text_models"]["other_models"]
    )
    return gpu_model_list, cpu_model_list

def assign_models_to_nodes(nodes: List[Dict], models: List[Dict], max_models: int, usage_data: List[Dict], db: firestore.Client) -> List[NodeAssignment]:
    assignments = []
    high_demand = HIGH_DEMAND_MODELS
    assigned_models = set()
    
    model_priority = sorted(
        models,
        key=lambda x: (
            x["name"] in high_demand,
            next((u["request_count"] for u in usage_data if u["model_name"] == x["name"]), 0),
            next((u["last_requested"] for u in usage_data if u["model_name"] == x["name"]), 0)
        ),
        reverse=True
    )

    is_gpu = nodes and nodes[0]["compute_resources"][0]["resource_type"] == "GPU"

    for model in model_priority:
        if model["name"] in assigned_models:
            continue
        is_high_demand = model["name"] in high_demand
        redundancy = MIN_REDUNDANCY if is_high_demand else 1
        
        for _ in range(redundancy):
            for node in nodes:
                if node["compute_resources"][0]["id"] in [a.node_id for a in assignments if len(a.models) >= max_models]:
                    continue
                node_resources = node["compute_resources"][0]
                available_vram = float(node_resources["gpu_specs"]["memory_size"].replace("GB", "")) if node_resources["resource_type"] == "GPU" else float("inf")
                available_ram = float(node_resources["ram"].replace("GB", ""))
                available_storage = float(node_resources["storage"]["capacity"].replace("GB", ""))
                
                used_vram = 0
                used_ram = 0
                used_storage = 0
                for a in assignments:
                    if a.node_id != node["compute_resources"][0]["id"]:
                        continue
                    for ma in a.models:
                        try:
                            m = next(mm for mm in models if mm["name"] == ma.model)
                            resources = estimate_model_resources(m, is_gpu)
                            used_vram += resources["vram"]
                            used_ram += resources["ram"]
                            used_storage += resources["storage"]
                        except StopIteration:
                            continue

                resources = estimate_model_resources(model, is_gpu)
                if (used_vram + resources["vram"] <= available_vram and
                    used_ram + resources["ram"] <= available_ram and
                    used_storage + resources["storage"] <= available_storage):
                    existing = next((a for a in assignments if a.node_id == node["compute_resources"][0]["id"]), None)
                    if existing and len(existing.models) < max_models:
                        existing.models.append(ModelAssignment(model=model["name"], hot=is_high_demand))
                    elif not existing:
                        assignments.append(NodeAssignment(
                            node_id=node["compute_resources"][0]["id"],
                            models=[ModelAssignment(model=model["name"], hot=is_high_demand)]
                        ))
                    assigned_models.add(model["name"])
                    db.collection("node_metrics").document(node["compute_resources"][0]["id"]).set(
                        {"request_count": firestore.Increment(1)}, merge=True
                    )
                    break

    return assignments

def find_best_node_for_model(nodes: List[Dict], model: Dict, is_gpu: bool, hot: bool, db: firestore.Client) -> List[Dict]:
    """
    Find the best 3 nodes for deploying a model based on resource availability and utilization.
    
    Args:
        nodes: List[Dict], list of node dictionaries from 'nodes' collection.
        model: Dict, model details including parameters.
        is_gpu: bool, whether the model requires GPU.
        hot: bool, whether the model should be deployed as hot.
        db: Firestore client instance.
    
    Returns:
        List[Dict], list of up to 3 dictionaries containing 'node_id', 'model', and 'hot' for the best nodes.
    """
    suitable_nodes = []
    resources = estimate_model_resources(model, is_gpu)
    print(f"Extracting the best nodes from the list of nodes")

    for node in nodes:
        # Validate compute_resources structure
        if not isinstance(node.get("compute_resources"), list) or not node["compute_resources"]:
            print(f"Skipping node {node.get('node_id', 'unknown')}: Invalid compute_resources structure")
            continue

        node_resources = node["compute_resources"][0]
        
        # Validate required fields
        if "resource_type" not in node_resources:
            print(f"Skipping node {node.get('node_id', 'unknown')}: Missing resource_type")
            continue

        # Validate gpu_specs for GPU nodes
        if node_resources["resource_type"] == "GPU":
            if not isinstance(node_resources.get("gpu_specs"), dict):
                print(f"Skipping node {node.get('node_id', 'unknown')}: gpu_specs is not a dictionary, got {type(node_resources.get('gpu_specs'))}")
                continue
            if "memory_size" not in node_resources["gpu_specs"]:
                print(f"Skipping node {node.get('node_id', 'unknown')}: Missing memory_size in gpu_specs")
                continue
            available_vram = parse_memory(node_resources["gpu_specs"]["memory_size"])
        else:
            available_vram = float("inf")

        # Validate ram
        if not isinstance(node_resources.get("ram"), str):
            print(f"Skipping node {node.get('node_id', 'unknown')}: ram is not a string, got {type(node_resources.get('ram'))}")
            continue
        available_ram = parse_memory(node_resources["ram"])

        # Validate storage
        if not isinstance(node_resources.get("storage"), dict):
            print(f"Skipping node {node.get('node_id', 'unknown')}: storage is not a dictionary, got {type(node_resources.get('storage'))}")
            continue
        if "capacity" not in node_resources["storage"]:
            print(f"Skipping node {node.get('node_id', 'unknown')}: Missing capacity in storage")
            continue
        available_storage = parse_memory(node_resources["storage"]["capacity"])

        metrics = get_node_utilization(node["node_id"], node_resources["resource_type"])
        utilization = metrics["gpu_utilization" if is_gpu else "cpu_utilization"]

        if (utilization <= UTILIZATION_THRESHOLD_HIGH and
            resources["vram"] <= available_vram and
            resources["ram"] <= available_ram and
            resources["storage"] <= available_storage):
            suitable_nodes.append({
                "node_id": node["node_id"],
                "model": model["name"],
                "hot": hot,
                "utilization": utilization,
                "memory": available_vram if is_gpu else available_ram,
                "uptime_seconds": node.get("uptime_seconds", 0)
            })
            db.collection("node_metrics").document(node["node_id"]).set({
                "cpu_utilization": metrics["cpu_utilization"],
                "gpu_utilization": metrics["gpu_utilization"],
                "request_count": firestore.Increment(1)
            }, merge=True)

    # Sort by utilization (ascending), memory (descending), and uptime (ascending)
    suitable_nodes.sort(key=lambda x: (x["utilization"], -x["memory"], x["uptime_seconds"]))
    print(f"Suitable nodes now available: {len(suitable_nodes)} nodes found")

    # Return the top 3 nodes (or fewer if less than 3 are available)
    return [{"node_id": node["node_id"], "model": node["model"], "hot": node["hot"]} for node in suitable_nodes[:1]]

def parse_memory(memory_str: str) -> float:
    """
    Parse a memory string (e.g., '10GB', '10TB') and convert to GB as a float.
    
    Args:
        memory_str: str, memory value with unit (e.g., '10GB', '10TB').
    
    Returns:
        float, memory value in GB.
    
    Raises:
        ValueError: If the memory string format is invalid.
    """
    if not memory_str or not isinstance(memory_str, str):
        return 0.0
    
    # Extract numeric part and unit using regex
    match = re.match(r'(\d+\.?\d*)\s*(GB|TB)?', memory_str.upper())
    if not match:
        raise ValueError(f"Invalid memory format: {memory_str}")
    
    value = float(match.group(1))
    unit = match.group(2)
    
    # Convert to GB
    if unit == "TB":
        return value * 1024
    return value

def extract_ssh_from_node(node: Dict) -> List[Dict]:
    """
    Extract the SSH information from the compute_resources of a single node dictionary.
    
    Args:
        node: Dict, node dictionary containing 'node_id' and 'compute_resources'.
    
    Returns:
        List[Dict], list containing a single dictionary with 'node_id' and 'ssh'.
    
    Raises:
        KeyError: If 'node_id' is missing in the node dictionary.
    """
    ssh_results = []
    
    logger.debug(f"Extracting SSH info from node: {node.get('node_id', 'unknown')}")

    node_id = node.get("node_id")
    if not node_id:
        logger.error(f"Missing node_id in node dictionary: {node}")
        raise KeyError("Missing node_id in node dictionary")

    try:
        # Extract SSH from compute_resources[0].network.ssh
        compute_resources = node.get("compute_resources", [])
        if not compute_resources:
            logger.warning(f"No compute_resources found for node {node_id}")
            return ssh_results

        network = compute_resources[0].get("network", {})
        ssh = network.get("ssh")
        if ssh:
            ssh_results.append({"node_id": node_id, "ssh": ssh})
            logger.info(f"Extracted SSH for node {node_id}: {ssh}")
        else:
            logger.warning(f"No SSH information found in network for node {node_id}: {network}")
    except (KeyError, IndexError, AttributeError) as e:
        logger.error(f"Error extracting SSH for node {node_id}: {str(e)}")

    if not ssh_results:
        logger.warning(f"No SSH information extracted for node {node_id}")

    logger.debug(f"SSH extraction complete. Results: {ssh_results}")
    return ssh_results

def deploy_model(node: Dict, model: Dict, is_hot: bool) -> str:
    ssh_info = node["compute_resources"][0]["network"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        if ssh_info["auth_type"] == "public_key":
            client.connect(
                hostname=ssh_info["internal_ip"],
                port=int(ssh_info["ssh"].split(":")[-1]),
                username=ssh_info["username"],
                key_filename="path_to_private_key"
            )
        else:
            client.connect(
                hostname=ssh_info["internal_ip"],
                port=int(ssh_info["ssh"].split(":")[-1]),
                username=ssh_info["username"],
                password=ssh_info["password"]
            )
        
        cmd = f"llama_cpp --model {model['model_id']} {'--hot' if is_hot else '--cold'}"
        stdin, stdout, stderr = client.exec_command(cmd)
        return f"Deployed {model['name']} (hot={is_hot}) to {node['compute_resources'][0]['id']}: {stdout.read().decode()}"
    except Exception as e:
        return f"Failed to deploy {model['name']} to {node['compute_resources'][0]['id']}: {str(e)}"
    finally:
        client.close()


def get_model_installations(model_id: str) -> ModelInstallationsResponse:
    try:
        # Fetch running installations for the model_id
        installations = db.collection("installations").where("model_id", "==", model_id).where("status", "==", "running").stream()
        installation_metrics = []
        total_chat_completions_hits = 0
        total_health_hits = 0
        chat_completions_latencies = []
        health_latencies = []

        for install in installations:
            install_data = install.to_dict()
            installation_id = install_data.get("installation_id")
            endpoint = install_data.get("endpoint")

            if not installation_id:
                continue

            # Fetch metrics for this installation
            metrics_doc = db.collection("metrics").document(installation_id).get()
            metrics_data = metrics_doc.to_dict() if metrics_doc.exists else {}
            metrics = metrics_data.get("metrics", {})

            # Extract /v1/chat/completions metrics
            chat_completions = metrics.get("/v1/chat/completions", {})
            chat_metrics = EndpointMetrics(
                avg_latency_ms=chat_completions.get("avg_latency_ms"),
                error_count=chat_completions.get("error_count"),
                hits=chat_completions.get("hits"),
                request_rate=chat_completions.get("request_rate"),
                success_count=chat_completions.get("success_count")
            )

            # Extract /health metrics
            health = metrics.get("/health", {})
            health_metrics = EndpointMetrics(
                avg_latency_ms=health.get("avg_latency_ms"),
                error_count=health.get("error_count"),
                hits=health.get("hits"),
                request_rate=health.get("request_rate"),
                success_count=health.get("success_count")
            )

            # Extract system metrics
            system = metrics.get("system", {})
            system_metrics = SystemMetrics(
                cpu_percent=system.get("cpu_percent"),
                memory_percent=system.get("memory_percent"),
                memory_total_mb=system.get("memory_total_mb"),
                memory_used_mb=system.get("memory_used_mb"),
                timestamp=system.get("timestamp")
            )

            # Update aggregate metrics
            if chat_metrics.hits:
                total_chat_completions_hits += chat_metrics.hits
                if chat_metrics.avg_latency_ms:
                    chat_completions_latencies.append(chat_metrics.avg_latency_ms)

            if health_metrics.hits:
                total_health_hits += health_metrics.hits
                if health_metrics.avg_latency_ms:
                    health_latencies.append(health_metrics.avg_latency_ms)

            # Add installation metrics to response
            installation_metrics.append(InstallationMetrics(
                installation_id=installation_id,
                endpoint=endpoint,
                chat_completions=chat_metrics,
                health=health_metrics,
                system=system_metrics
            ))

        # Calculate average latencies
        avg_chat_completions_latency = (
            sum(chat_completions_latencies) / len(chat_completions_latencies)
            if chat_completions_latencies else None
        )
        avg_health_latency = (
            sum(health_latencies) / len(health_latencies)
            if health_latencies else None
        )

        return ModelInstallationsResponse(
            model_id=model_id,
            total_installations=len(installation_metrics),
            total_chat_completions_hits=total_chat_completions_hits,
            total_health_hits=total_health_hits,
            avg_chat_completions_latency_ms=avg_chat_completions_latency,
            avg_health_latency_ms=avg_health_latency,
            installations=installation_metrics
        )
    except Exception as e:
        return f"Error retrieving model installations metrics: {str(e)}"