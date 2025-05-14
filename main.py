from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from utility.utils import load_data_from_db, get_available_nodes, flatten_models, assign_models_to_nodes, find_best_node_for_model, deploy_model,update_high_demand_models
from db.database import FirestoreDB
from scheduler.scheduler import Scheduler
import asyncio
import threading
from models import *
from collections import defaultdict
from dateutil import parser
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models (unchanged)

# Initialize FastAPI app
app = FastAPI(title="POLARIS SCHEDULER")
SCHEDULER_INTERVAL = 300

# Initialize Firestore
db = FirestoreDB("db/creds.json")

# Load data from Firestore
gpu_models, cpu_models, nodes = load_data_from_db(db.db)
gpu_model_list, cpu_model_list = flatten_models(gpu_models, cpu_models)
# Lock for file operations (kept for potential future use)
file_lock = threading.Lock()

# Initialize scheduler
scheduler = Scheduler(db, float(SCHEDULER_INTERVAL))

async def get_installations_data(db, model_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch installations data for given model IDs, only including installations with status 'running'.
    Returns: Dict[model_id, {"count": int, "public_urls": List[str], "has_hot": bool, "installations": List[Dict]}]
    """
    installations_ref = db.collection("installations")
    installations_data = defaultdict(lambda: {"count": 0, "public_urls": [], "has_hot": False, "installations": []})

    for i in range(0, len(model_ids), 10):
        batch_ids = model_ids[i:i+10]
        query = installations_ref.where("model_id", "in", batch_ids).where("status", "==", "running")
        for doc in query.stream():
            data = doc.to_dict()
            model_id = data["model_id"]
            installations_data[model_id]["count"] += 1
            if data.get("public_url"):
                installations_data[model_id]["public_urls"].append(data["public_url"])
            if data.get("status") == "running":
                installations_data[model_id]["has_hot"] = True
            # Add installation details
            installations_data[model_id]["installations"].append({
                "installation_id": data.get("installation_id"),
                "endpoint": data.get("endpoint"),
                "public_url": data.get("public_url"),
                "status": data.get("status")
            })

    return installations_data

async def get_metrics_data(db, installation_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch metrics data for given installation IDs from the metrics collection.
    Returns: Dict[installation_id, Dict]
    """
    metrics_data = {}
    for installation_id in installation_ids:
        doc = db.db.collection("metrics").document(installation_id).get()
        if doc.exists:
            metrics_data[installation_id] = doc.to_dict().get("metrics", {})
    return metrics_data

@app.on_event("startup")
async def startup_event():
    # asyncio.create_task(scheduler.run())
    pass

@app.get("/installations", response_model=List[InstallationMetrics])
async def get_installations(installation_id: Optional[str] = None):
    try:
        # Fetch installations from Firestore
        installations_ref = db.db.collection("installations")
        if installation_id:
            doc = installations_ref.document(installation_id).get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail=f"Installation {installation_id} not found")
            installations = [doc.to_dict()]
            logger.info(f"Fetched installation {installation_id}: {installations[0]}")
        else:
            installations = [doc.to_dict() for doc in installations_ref.stream()]
            logger.info(f"Fetched {len(installations)} installations")

        # Extract installation IDs for metrics lookup
        installation_ids = [install.get("installation_id", "") for install in installations]
        metrics_data = await get_metrics_data(db, installation_ids) if installation_ids else {}

        # Build InstallationMetrics list
        installations_metrics = []
        for install in installations:
            installation_id = install.get("installation_id", "")
            metrics = metrics_data.get(installation_id, {})
            chat_completions = metrics.get("/v1/chat/completions", {})
            health = metrics.get("/health", {})
            system = metrics.get("system", {})

            installation_metrics = InstallationMetrics(
                installation_id=installation_id,
                endpoint=install.get("endpoint", ""),
                model=install.get("model", ""),  # New field
                status=install.get("status", ""),  # New field
                public_url=install.get("public_url", ""),  # New field
                model_id=install.get("model_id", ""),  # New field
                chat_completions=EndpointMetrics(
                    avg_latency_ms=chat_completions.get("avg_latency_ms"),
                    error_count=chat_completions.get("error_count"),
                    hits=chat_completions.get("hits"),
                    request_rate=chat_completions.get("request_rate"),
                    success_count=chat_completions.get("success_count")
                ),
                health=EndpointMetrics(
                    avg_latency_ms=health.get("avg_latency_ms"),
                    error_count=health.get("error_count"),
                    hits=health.get("hits"),
                    request_rate=health.get("request_rate"),
                    success_count=health.get("success_count")
                ),
                system=SystemMetrics(
                    cpu_percent=system.get("cpu_percent"),
                    memory_percent=system.get("memory_percent"),
                    memory_total_mb=system.get("memory_total_mb"),
                    memory_used_mb=system.get("memory_used_mb"),
                    timestamp=system.get("timestamp")
                )
            )
            installations_metrics.append(installation_metrics)

        logger.info(f"Built {len(installations_metrics)} installation metrics objects")
        return installations_metrics
    except Exception as e:
        logger.error(f"Error retrieving installations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving installations: {str(e)}")
@app.get("/model-installations/{model_id}", response_model=ModelInstallationsResponse)
async def get_model_installations_metrics(model_id: str):
    try:
        gpu_doc = db.db.collection("gpu_models").where("model_id", "==", model_id).get()
        cpu_doc = db.db.collection("cpu_models").where("model_id", "==", model_id).get()
        if not gpu_doc and not cpu_doc:
            raise HTTPException(status_code=404, detail=f"Model ID {model_id} not found")

        # Fetch running installations for the model_id
        installations = db.db.collection("installations").where("model_id", "==", model_id).where("status", "==", "running").stream()
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
            metrics_doc = db.db.collection("metrics").document(installation_id).get()
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
        raise HTTPException(status_code=500, detail=f"Error retrieving model installations metrics: {str(e)}")

@app.get("/installation-status-summary", response_model=InstallationStatusSummary)
async def get_installation_status_summary():
    try:
        # Query all installations
        installations = db.db.collection("installations").stream()
        total_installations = 0
        total_running_installations = 0
        total_stopped_installations = 0

        for install in installations:
            install_data = install.to_dict()
            total_installations += 1
            status = install_data.get("status", "").lower()
            if status == "running":
                total_running_installations += 1
            elif status == "stopped":
                total_stopped_installations += 1

        logger.info(f"Installation Summary - Total: {total_installations}, Running: {total_running_installations}, Stopped: {total_stopped_installations}")

        return InstallationStatusSummary(
            total_installations=total_installations,
            total_running_installations=total_running_installations,
            total_stopped_installations=total_stopped_installations
        )

    except Exception as e:
        logger.error(f"Error retrieving installation status summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving installation status summary: {str(e)}")
    
@app.post("/high-demand-models", response_model=HighDemandModelRequest)
async def add_high_demand_model(request: HighDemandModelRequest):
    try:
        # Validate model exists in gpu_models or cpu_models
        gpu_doc = db.db.collection("gpu_models").document(request.model_name).get()
        cpu_doc = db.db.collection("cpu_models").document(request.model_name).get()
        if not gpu_doc.exists and not cpu_doc.exists:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found in GPU or CPU models")

        # Get current high-demand models
        doc = db.db.collection("high_demand_models").document("config").get()
        current_models = doc.to_dict().get("models", []) if doc.exists else []

        # Add model if not already present
        if request.model_name not in current_models:
            current_models.append(request.model_name)
            update_high_demand_models(db.db, current_models)
        
        return {"model_name": request.model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding high-demand model: {str(e)}")

@app.delete("/high-demand-models/{model_name}")
async def delete_high_demand_model(model_name: str):
    try:
        # Get current high-demand models
        doc = db.db.collection("high_demand_models").document("config").get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="No high-demand models found")

        current_models = doc.to_dict().get("models", [])
        if model_name not in current_models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not in high-demand models")

        # Remove model
        current_models.remove(model_name)
        update_high_demand_models(db.db, current_models)

        return {"message": f"Model {model_name} removed from high-demand models"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting high-demand model: {str(e)}")

@app.get("/high-demand-models", response_model=List[str])
async def get_high_demand_models():
    try:
        doc = db.db.collection("high_demand_models").document("config").get()
        if not doc.exists:
            return []
        return doc.to_dict().get("models", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving high-demand models: {str(e)}")

# Existing endpoints (unchanged exce

# GPU Model Endpoints
@app.post("/gpu-model", response_model=Model)
async def add_gpu_model(model: Model):
    try:
        valid_types = ["vision", "vision-video", "vision-video-audio", "audio","instruct","chat"]
        if model.type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid model type. Must be one of {valid_types}")

        # Check if model exists
        doc = db.db.collection("gpu_models").document(model.name).get()
        if doc.exists:
            raise HTTPException(status_code=400, detail=f"Model {model.name} already exists")

        # Save to Firestore
        db.db.collection("gpu_models").document(model.name).set(model.dict(exclude_none=True))

        # Update global data
        global gpu_model_list, gpu_models
        gpu_models["multimodal_models"]["other_models"].append(model.dict(exclude_none=True))
        gpu_model_list, _ = flatten_models(gpu_models, cpu_models)

        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding GPU model: {str(e)}")

@app.get("/gpu-models", response_model=List[Model])
async def get_gpu_models(model_name: Optional[str] = None):
    try:
        # Fetch all GPU models from Firestore
        gpu_models_ref = db.db.collection("gpu_models").stream()
        models = [doc.to_dict() for doc in gpu_models_ref]
        logger.info(f"Fetched {len(models)} GPU models from Firestore")

        if model_name:
            model = next((m for m in models if m.get("name") == model_name), None)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            models = [model]

        # Fetch installations data for all model IDs
        model_ids = [m["model_id"] for m in models]
        installations_data = await get_installations_data(db.db, model_ids)
        logger.info(f"Fetched installations data for {len(model_ids)} model IDs: {installations_data}")

        # Extract installation IDs for metrics lookup
        installation_ids = [
            install["installation_id"]
            for model_id in model_ids
            for install in installations_data.get(model_id, {}).get("installations", [])
        ]
        metrics_data = await get_metrics_data(db, installation_ids) if installation_ids else {}
        logger.info(f"Fetched metrics data for {len(installation_ids)} installations: {metrics_data}")

        # Enrich models with installations and metrics data
        enriched_models = []
        for model in models:
            model_id = model["model_id"]
            install_info = installations_data.get(model_id, {
                "count": 0,
                "public_urls": [],
                "has_hot": False,
                "installations": []
            })
            installation_metrics_list = []

            for install in install_info["installations"]:
                installation_id = install.get("installation_id", "")
                metrics = metrics_data.get(installation_id, {})
                chat_completions = metrics.get("/v1/chat/completions", {})
                health = metrics.get("/health", {})
                system = metrics.get("system", {})

                # Append metrics to installation
                installation_metrics = InstallationMetrics(
                    installation_id=installation_id,
                    endpoint=install.get("endpoint", ""),
                    chat_completions=EndpointMetrics(
                        avg_latency_ms=chat_completions.get("avg_latency_ms"),
                        error_count=chat_completions.get("error_count"),
                        hits=chat_completions.get("hits"),
                        request_rate=chat_completions.get("request_rate"),
                        success_count=chat_completions.get("success_count")
                    ),
                    health=EndpointMetrics(
                        avg_latency_ms=health.get("avg_latency_ms"),
                        error_count=health.get("error_count"),
                        hits=health.get("hits"),
                        request_rate=health.get("request_rate"),
                        success_count=health.get("success_count")
                    ),
                    system=SystemMetrics(
                        cpu_percent=system.get("cpu_percent"),
                        memory_percent=system.get("memory_percent"),
                        memory_total_mb=system.get("memory_total_mb"),
                        memory_used_mb=system.get("memory_used_mb"),
                        timestamp=system.get("timestamp")
                    )
                )
                installation_metrics_list.append(installation_metrics)

            logger.info(f"Installation metrics for model {model_id}: {installation_metrics_list}")

            # Create a new dictionary to avoid keyword argument conflicts
            enriched_model_data = {
                **{k: v for k, v in model.items() if k in Model.__fields__.keys()},
                "installations_count": len(install_info["installations"]),
                "public_urls": install_info["public_urls"],
                "status": "hot" if install_info["has_hot"] else "cold",
                "installation_metrics": installation_metrics_list
            }
            model_instance = Model(**enriched_model_data)
            enriched_models.append(model_instance.dict())

        # Sort by number of public URLs (descending)
        enriched_models.sort(key=lambda x: len(x["public_urls"]), reverse=True)

        return [Model(**m) for m in enriched_models]
    except Exception as e:
        logger.error(f"Error retrieving GPU models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving GPU models: {str(e)}")

@app.put("/gpu-model/{model_name}", response_model=Model)
async def update_gpu_model(model_name: str, model: Model):
    try:
        valid_types = ["vision", "vision-video", "vision-video-audio", "audio"]
        if model.type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid model type. Must be one of {valid_types}")

        doc = db.db.collection("gpu_models").document(model_name).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        db.db.collection("gpu_models").document(model_name).set(model.dict(exclude_none=True))

        global gpu_model_list, gpu_models
        for i, m in enumerate(gpu_models["multimodal_models"]["other_models"]):
            if m["name"] == model_name:
                gpu_models["multimodal_models"]["other_models"][i] = model.dict(exclude_none=True)
                break
        gpu_model_list, _ = flatten_models(gpu_models, cpu_models)

        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating GPU model: {str(e)}")

@app.delete("/gpu-model/{model_name}")
async def delete_gpu_model(model_name: str):
    try:
        doc = db.db.collection("gpu_models").document(model_name).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        db.db.collection("gpu_models").document(model_name).delete()

        global gpu_model_list, gpu_models
        gpu_models["multimodal_models"]["other_models"] = [
            m for m in gpu_models["multimodal_models"]["other_models"] if m["name"] != model_name
        ]
        gpu_model_list, _ = flatten_models(gpu_models, cpu_models)

        return {"message": f"Model {model_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting GPU model: {str(e)}")

# CPU Model Endpoints
@app.post("/cpu-model", response_model=Model)
async def add_cpu_model(model: Model):
    try:
        valid_types = ["instruct", "chat", "coding", "math", "multilingual"]
        if model.type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid model type. Must be one of {valid_types}")

        if not model.quantization:
            raise HTTPException(status_code=400, detail="Quantization list is required for CPU models")

        doc = db.db.collection("cpu_models").document(model.name).get()
        if doc.exists:
            raise HTTPException(status_code=400, detail=f"Model {model.name} already exists")

        db.db.collection("cpu_models").document(model.name).set(model.dict())

        global cpu_model_list, cpu_models
        if model.type in ["coding", "math", "multilingual"]:
            category = f"{model.type}_models"
            cpu_models["specialized_models"][category].append(model.dict())
        elif model.parameters in ["1.1B", "1B", "3B"] and model.type in ["chat", "instruct"]:
            cpu_models["lightweight_models"]["tiny_models"].append(model.dict())
        else:
            cpu_models["text_models"]["other_models"].append(model.dict())
        _, cpu_model_list = flatten_models(gpu_models, cpu_models)

        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding CPU model: {str(e)}")

@app.get("/cpu-models", response_model=List[Model])
async def get_cpu_models(model_name: Optional[str] = None):
    try:
        # Fetch all CPU models from Firestore
        cpu_models_ref = db.db.collection("cpu_models").stream()
        models = [doc.to_dict() for doc in cpu_models_ref]
        logger.info(f"Fetched {len(models)} CPU models from Firestore")

        if model_name:
            model = next((m for m in models if m.get("name") == model_name), None)
            if not model:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            models = [model]

        # Fetch installations data for all model IDs
        model_ids = [m["model_id"] for m in models]
        installations_data = await get_installations_data(db.db, model_ids)
        logger.info(f"Fetched installations data for {len(model_ids)} model IDs")

        # Extract installation IDs for metrics lookup
        installation_ids = [
            install["installation_id"]
            for model_id in model_ids
            for install in installations_data.get(model_id, {}).get("installations", [])
        ]
        metrics_data = await get_metrics_data(db, installation_ids) if installation_ids else {}
        logger.info(f"Fetched metrics data for {len(installation_ids)} installations: {metrics_data}")

        # Enrich models with installations and metrics data
        enriched_models = []
        for model in models:
            model_id = model["model_id"]
            install_info = installations_data.get(model_id, {"installations": [], "has_hot": False, "public_urls": [], "count": 0})
            installation_metrics_list = []

            for install in install_info["installations"]:
                installation_id = install.get("installation_id", "")
                metrics = metrics_data.get(installation_id, {})
                chat_completions = metrics.get("/v1/chat/completions", {})
                health = metrics.get("/health", {})
                system = metrics.get("system", {})

                # Append metrics to installation
                installation_metrics = InstallationMetrics(
                    installation_id=installation_id,
                    endpoint=install.get("endpoint", ""),
                    chat_completions=EndpointMetrics(
                        avg_latency_ms=chat_completions.get("avg_latency_ms"),
                        error_count=chat_completions.get("error_count"),
                        hits=chat_completions.get("hits"),
                        request_rate=chat_completions.get("request_rate"),
                        success_count=chat_completions.get("success_count")
                    ),
                    health=EndpointMetrics(
                        avg_latency_ms=health.get("avg_latency_ms"),
                        error_count=health.get("error_count"),
                        hits=health.get("hits"),
                        request_rate=health.get("request_rate"),
                        success_count=health.get("success_count")
                    ),
                    system=SystemMetrics(
                        cpu_percent=system.get("cpu_percent"),
                        memory_percent=system.get("memory_percent"),
                        memory_total_mb=system.get("memory_total_mb"),
                        memory_used_mb=system.get("memory_used_mb"),
                        timestamp=system.get("timestamp")
                    )
                )
                installation_metrics_list.append(installation_metrics)

            # Create a new dictionary to avoid keyword argument conflicts
            enriched_model_data = {
                **{k: v for k, v in model.items() if k in Model.__fields__.keys()},
                "installations_count": len(install_info["installations"]),
                "public_urls": [install["public_url"] for install in install_info["installations"] if install.get("public_url")],
                "status": "hot" if install_info["has_hot"] else "cold",
                "installation_metrics": installation_metrics_list
            }
            model_instance = Model(**enriched_model_data)
            enriched_models.append(model_instance.dict())  # Removed exclude_none=True

        # Sort by number of public URLs (descending)
        enriched_models.sort(key=lambda x: len(x["public_urls"]), reverse=True)

        return [Model(**m) for m in enriched_models]
    except Exception as e:
        logger.error(f"Error retrieving CPU models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving CPU models: {str(e)}")

@app.put("/cpu-model/{model_name}", response_model=Model)
async def update_cpu_model(model_name: str, model: Model):
    try:
        valid_types = ["instruct", "chat", "coding", "math", "multilingual"]
        if model.type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid model type. Must be one of {valid_types}")

        if not model.quantization:
            raise HTTPException(status_code=400, detail="Quantization list is required for CPU models")

        doc = db.db.collection("cpu_models").document(model_name).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        db.db.collection("cpu_models").document(model_name).set(model.dict())

        global cpu_model_list, cpu_models
        for section in ["text_models", "specialized_models", "lightweight_models"]:
            for subcategory in cpu_models[section]:
                for i, m in enumerate(cpu_models[section][subcategory]):
                    if m["name"] == model_name:
                        cpu_models[section][subcategory][i] = model.dict()
                        break
        _, cpu_model_list = flatten_models(gpu_models, cpu_models)

        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating CPU model: {str(e)}")

@app.delete("/cpu-model/{model_name}")
async def delete_cpu_model(model_name: str):
    try:
        doc = db.db.collection("cpu_models").document(model_name).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        db.db.collection("cpu_models").document(model_name).delete()

        global cpu_model_list, cpu_models
        for section in ["text_models", "specialized_models", "lightweight_models"]:
            for subcategory in cpu_models[section]:
                cpu_models[section][subcategory] = [
                    m for m in cpu_models[section][subcategory] if m["name"] != model_name
                ]
        _, cpu_model_list = flatten_models(gpu_models, cpu_models)

        return {"message": f"Model {model_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting CPU model: {str(e)}")

# Node Endpoints
@app.post("/node", response_model=Node)
async def add_node(node: Node, background_tasks: BackgroundTasks):
    try:
        node_id = node.compute_resources[0]["id"]
        doc = db.db.collection("nodes").document(node_id).get()
        if doc.exists:
            raise HTTPException(status_code=400, detail=f"Node {node_id} already exists")

        db.db.collection("nodes").document(node_id).set(node.dict())

        global nodes
        nodes[node_id] = node.dict()

        return node
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding node: {str(e)}")

@app.get("/nodes", response_model=List[Node])
async def get_nodes(node_id: Optional[str] = None):
    try:
        if node_id:
            doc = db.collection("nodes").document(node_id).get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
            return [Node(**doc.to_dict())]
        nodes_list = [Node(**doc.to_dict()) for doc in db.db.collection("nodes").stream()]
        return nodes_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving nodes: {str(e)}")

@app.put("/node/{node_id}", response_model=Node)
async def update_node(node_id: str, node: Node, background_tasks: BackgroundTasks):
    try:
        doc = db.db.collection("nodes").document(node_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        db.db.collection("nodes").document(node_id).set(node.dict())

        global nodes
        nodes[node_id] = node.dict()

        return node
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating node: {str(e)}")

@app.delete("/node/{node_id}")
async def delete_node(node_id: str, background_tasks: BackgroundTasks):
    try:
        doc = db.db.collection("nodes").document(node_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

        db.db.collection("nodes").document(node_id).delete()

        global nodes
        if node_id in nodes:
            del nodes[node_id]

        return {"message": f"Node {node_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting node: {str(e)}")

# Assignment Endpoints
@app.get("/auto-match", response_model=AssignmentResponse)
async def auto_match_models(background_tasks: BackgroundTasks):
    try:
        gpu_nodes = get_available_nodes(nodes, "GPU", [], db.db)
        cpu_nodes = get_available_nodes(nodes, "CPU", [], db.db)
        gpu_assignments = assign_models_to_nodes(gpu_nodes, gpu_model_list, max_models=3, usage_data=[], db=db.db)
        cpu_assignments = assign_models_to_nodes(cpu_nodes, cpu_model_list, max_models=2, usage_data=[], db=db.db)

        for assignment in gpu_assignments + cpu_assignments:
            node = next(n for n in nodes.values() if n["compute_resources"][0]["id"] == assignment.node_id)
            for model_assignment in assignment.models:
                model = next(m for m in (gpu_model_list + cpu_model_list) if m["name"] == model_assignment.model)
                background_tasks.add_task(deploy_model, node, model, model_assignment.hot)

        db.save_assignments({"gpu": gpu_assignments, "cpu": cpu_assignments})

        return AssignmentResponse(
            gpu_assignments=[
                Assignment(
                    node_id=a.node_id,
                    type="gpu",
                    models=[{"model": m.model, "hot": m.hot} for m in a.models]
                ) for a in gpu_assignments
            ],
            cpu_assignments=[
                Assignment(
                    node_id=a.node_id,
                    type="cpu",
                    models=[{"model": m.model, "hot": m.hot} for m in a.models]
                ) for a in cpu_assignments
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assigning models: {str(e)}")



@app.get("/assignments", response_model=AssignmentResponse)
async def get_assignments(node_id: Optional[str] = None):
    try:
        assignments = db.get_assignments()
        if node_id:
            gpu_assignment = next((a for a in assignments["gpu_assignments"] if a["node_id"] == node_id), None)
            cpu_assignment = next((a for a in assignments["cpu_assignments"] if a["node_id"] == node_id), None)
            if not gpu_assignment and not cpu_assignment:
                raise HTTPException(status_code=404, detail=f"Assignment for node {node_id} not found")
            return AssignmentResponse(
                gpu_assignments=[Assignment(**gpu_assignment)] if gpu_assignment else [],
                cpu_assignments=[Assignment(**cpu_assignment)] if cpu_assignment else []
            )
        return AssignmentResponse(
            gpu_assignments=[Assignment(**a) for a in assignments["gpu_assignments"]],
            cpu_assignments=[Assignment(**a) for a in assignments["cpu_assignments"]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving assignments: {str(e)}")

@app.put("/assignment/{node_id}", response_model=Assignment)
async def update_assignment(node_id: str, assignment: Assignment, background_tasks: BackgroundTasks):
    try:
        if assignment.node_id != node_id:
            raise HTTPException(status_code=400, detail="Node ID in body must match URL parameter")
        if assignment.type not in ["gpu", "cpu"]:
            raise HTTPException(status_code=400, detail="Assignment type must be 'gpu' or 'cpu'")

        model_list = gpu_model_list if assignment.type == "gpu" else cpu_model_list
        for m in assignment.models:
            if not any(model["name"] == m["model"] for model in model_list):
                raise HTTPException(status_code=400, detail=f"Model {m['model']} not found in {assignment.type} models")

        doc = db.db.collection("assignments").document(node_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Assignment for node {node_id} not found")

        db.db.collection("assignments").document(node_id).set({
            "node_id": node_id,
            "type": assignment.type,
            "models": assignment.models,
            "timestamp": db.db.server_timestamp()
        }, merge=True)

        node = next((n for n in nodes.values() if n["compute_resources"][0]["id"] == node_id), None)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        for model_assignment in assignment.models:
            model = next(m for m in (gpu_model_list + cpu_model_list) if m["name"] == model_assignment["model"])
            background_tasks.add_task(deploy_model, node, model, model_assignment["hot"])

        return assignment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating assignment: {str(e)}")

@app.delete("/assignment/{node_id}")
async def delete_assignment(node_id: str):
    try:
        doc = db.db.collection("assignments").document(node_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Assignment for node {node_id} not found")

        db.db.collection("assignments").document(node_id).delete()
        return {"message": f"Assignment for node {node_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting assignment: {str(e)}")

@app.post("/deploy-model", response_model=SingleModelResponse)
async def match_single_model(request: SingleModelRequest, background_tasks: BackgroundTasks):
    try:
        model_list = gpu_model_list if request.model_type.lower() == "gpu" else cpu_model_list
        model = next((m for m in model_list if m["name"] == request.model_name), None)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
        resource_type = "GPU" if request.model_type.lower() == "gpu" else "CPU"
        nodes_list = get_available_nodes(nodes, resource_type, [], db.db)
        assignment = find_best_node_for_model(nodes_list, model, is_gpu=request.model_type.lower() == "gpu", hot=request.hot, db=db.db)
        if not assignment:
            raise HTTPException(status_code=404, detail=f"No suitable node found for model {request.model_name}")

        node = next(n for n in nodes.values() if n["compute_resources"][0]["id"] == assignment["node_id"])
        # background_tasks.add_task(deploy_model, node, model, request.hot)

        return SingleModelResponse(
            node_id=assignment["node_id"],
            model=assignment["model"],
            hot=request.hot
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error matching model: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)