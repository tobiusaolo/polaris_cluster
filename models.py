from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

class ModelAssignment(BaseModel):
    model: str
    hot: bool

class NodeAssignment(BaseModel):
    node_id: str
    models: List[ModelAssignment]



class ModelUsage(BaseModel):
    model_name: str
    request_count: int
    last_requested: float  # timestamp

class ComputeResource(BaseModel):
    id: str
    type: str
    specs: Dict[str, str]

class Node(BaseModel):
    status: str
    available: bool
    compute_resources: List[Dict]  # Note: Should be List[ComputeResource] for consistency
    last_checked: Optional[datetime] = None  # Changed to datetime
    uptime_seconds: Optional[float] = None
    hot: Optional[bool] = False

class Assignment(BaseModel):
    node_id: str
    type: str
    models: List[Dict]

class AssignmentResponse(BaseModel):
    gpu_assignments: List[Assignment]
    cpu_assignments: List[Assignment]

class SingleModelRequest(BaseModel):
    model_name: str
    model_type: str
    hot: bool = False


class HighDemandModelRequest(BaseModel):
    model_name: str

class EndpointMetrics(BaseModel):
    avg_latency_ms: Optional[float] = None
    error_count: Optional[int] = None
    hits: Optional[int] = None
    request_rate: Optional[float] = None
    success_count: Optional[int] = None

class SystemMetrics(BaseModel):
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_total_mb: Optional[float] = None
    memory_used_mb: Optional[float] = None
    timestamp: Optional[str] = None

class InstallationMetrics(BaseModel):
    installation_id: str
    endpoint: Optional[str] = None
    model: Optional[str] = None  # New field for model name
    status: Optional[str] = None  # New field for installation status
    public_url: Optional[str] = None  # New field for public URL
    model_id: Optional[str] = None  # New field for model ID
    chat_completions: Optional[EndpointMetrics] = None
    health: Optional[EndpointMetrics] = None
    system: Optional[SystemMetrics] = None

class ModelInstallationsResponse(BaseModel):
    model_id: str
    total_installations: int
    total_chat_completions_hits: int
    total_health_hits: int
    avg_chat_completions_latency_ms: Optional[float] = None
    avg_health_latency_ms: Optional[float] = None
    installations: List[InstallationMetrics]

class InstallationStatusSummary(BaseModel):
    total_installations: int
    total_running_installations: int
    total_stopped_installations: int


class Model(BaseModel):
    name: str
    model_id: str
    parameters: str
    type: str
    model_path: str
    requires: str
    description: str
    quantization: Optional[List[str]] = None
    installations_count: Optional[int] = 0  # Number of installations
    public_urls: Optional[List[str]] = []  # List of public URLs
    status: Optional[str] = "cold"
    installation_metrics: Optional[List[InstallationMetrics]] = None

class SingleModelResponse(BaseModel):
    node_id: str
    model: str
    hot: bool
    deployment_status: str  # New field: "success" or "failed"
    installations: Optional[List[InstallationMetrics]] = None


class ChatRequest(BaseModel):
    model_id: str

class ChatResponse(BaseModel):
    public_url: str
    installation_id: str
    model_id: str
    hits: int