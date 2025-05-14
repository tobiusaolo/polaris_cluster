import paramiko
from typing import List, Dict, Optional
from dataclasses import dataclass
from google.cloud import firestore
import re
import time

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

def load_data_from_db(db: firestore.Client) -> tuple[Dict, Dict, Dict]:
    """
    Load GPU models, CPU models, nodes, and high-demand models from Firestore.
    Returns: (gpu_models, cpu_models, nodes)
    """
    global HIGH_DEMAND_MODELS
    # Load GPU models
    gpu_models = {"multimodal_models": {
        "qwen_models": [],
        "deepseek_models": [],
        "llama_models": [],
        "internvl_models": [],
        "phi_models": [],
        "other_models": []
    }}
    for doc in db.collection("gpu_models").stream():
        model = doc.to_dict()
        gpu_models["multimodal_models"]["other_models"].append(model)

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

    # Load nodes
    nodes = {}
    for doc in db.collection("nodes").stream():
        nodes[doc.id] = doc.to_dict()

    # Load high-demand models
    doc = db.collection("high_demand_models").document("config").get()
    if doc.exists:
        HIGH_DEMAND_MODELS = set(doc.to_dict().get("models", []))
    else:
        # Initialize with empty list if not exists
        db.collection("high_demand_models").document("config").set({"models": []})
        HIGH_DEMAND_MODELS = set()

    return gpu_models, cpu_models, nodes

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

def get_node_utilization(node: Dict, resource_type: str) -> Dict:
    """Fetch real-time CPU/GPU utilization via SSH."""
    ssh_info = node["compute_resources"][0]["network"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        # if ssh_info["auth_type"] == "public_key":
        #     client.connect(
        #         hostname=ssh_info["internal_ip"],
        #         port=int(ssh_info["ssh"].split(":")[-1]),
        #         username=ssh_info["username"],
        #     )
        # else:
        #     client.connect(
        #         hostname=ssh_info["internal_ip"],
        #         port=int(ssh_info["ssh"].split(":")[-1]),
        #         username=ssh_info["username"],
        #         password=ssh_info["password"]
        #     )

        metrics = {"cpu_utilization": 0.0, "gpu_utilization": 0.0}
        if resource_type == "CPU":
            stdin, stdout, stderr = client.exec_command("top -bn1 | head -n 3")
            output = stdout.read().decode()
            cpu_match = re.search(r"%Cpu\(s\):\s*([\d.]+)\s*us", output)
            if cpu_match:
                metrics["cpu_utilization"] = float(cpu_match.group(1)) / 100.0
        else:
            stdin, stdout, stderr = client.exec_command("nvidia-smi --query-gpu=utilization.gpu --format=csv")
            output = stdout.read().decode()
            gpu_match = re.search(r"(\d+)%", output)
            if gpu_match:
                metrics["gpu_utilization"] = float(gpu_match.group(1)) / 100.0

        return metrics
    except Exception as e:
        print(f"Error fetching utilization for node {node['compute_resources'][0]['id']}: {str(e)}")
        return {"cpu_utilization": 0.0, "gpu_utilization": 0.0}
    finally:
        client.close()

def get_available_nodes(nodes: Dict, resource_type: str, usage_data: List[Dict], db: firestore.Client) -> List[Dict]:

    available = [
        node for node_id, node in nodes.items()
        # if node["status"] == "offline" and node["available"] and node["compute_resources"][0]["resource_type"] == resource_type]
        if node["status"] == "offline" and node["compute_resources"][0]["resource_type"] == resource_type]
    node_metrics = {}
    for node in available:
        metrics = get_node_utilization(node, resource_type)
        node_metrics[node["compute_resources"][0]["id"]] = metrics
        db.collection("node_metrics").document(node["compute_resources"][0]["id"]).set({
            "cpu_utilization": metrics["cpu_utilization"],
            "gpu_utilization": metrics["gpu_utilization"],
        }, merge=True)

    return sorted(
        available,
        key=lambda x: (
            node_metrics[x["compute_resources"][0]["id"]]["gpu_utilization" if resource_type == "GPU" else "cpu_utilization"],
            float(x["compute_resources"][0]["gpu_specs"]["memory_size"].replace("GB", "")) if resource_type == "GPU"
            else float(x["compute_resources"][0]["ram"].replace("GB", "")),
            x["uptime_seconds"]
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

def find_best_node_for_model(nodes: List[Dict], model: Dict, is_gpu: bool, hot: bool, db: firestore.Client) -> Optional[Dict]:
    for node in nodes:
        node_resources = node["compute_resources"][0]
        available_vram = float(node_resources["gpu_specs"]["memory_size"].replace("GB", "")) if node_resources["resource_type"] == "GPU" else float("inf")
        available_ram = float(node_resources["ram"].replace("GB", ""))
        available_storage = float(node_resources["storage"]["capacity"].replace("GB", ""))
        resources = estimate_model_resources(model, is_gpu)

        # metrics = get_node_utilization(node, node_resources["resource_type"])
        # utilization = metrics["gpu_utilization" if is_gpu else "cpu_utilization"]
        # if utilization > UTILIZATION_THRESHOLD_HIGH:
        #     continue
        

        if (resources["vram"] <= available_vram and
            resources["ram"] <= available_ram and
            resources["storage"] <= available_storage):
            db.collection("node_metrics").document(node["compute_resources"][0]["id"]).set({
                "cpu_utilization": metrics["cpu_utilization"],
                "gpu_utilization": metrics["gpu_utilization"],
                "request_count": firestore.Increment(1)
            }, merge=True)
            return {"node_id": node["compute_resources"][0]["id"], "model": model["name"], "hot": hot}
    return None

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