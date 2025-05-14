import firebase_admin
from firebase_admin import credentials, firestore
from typing import List, Dict, Optional
import time
from utility.utils import NodeAssignment, ModelAssignment  # Import dataclasses

class FirestoreDB:
    def __init__(self, credentials_path: str):
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.batch_size = 500  # Firestore batch limit

    def update_node_status(self, nodes: Dict):
        batch = self.db.batch()
        count = 0
        for node_id, node in nodes.items():
            ref = self.db.collection("nodes").document(node_id)
            batch.set(ref, {
                "status": node["status"],
                "available": node["available"],
                "compute_resources": node["compute_resources"],
                "last_checked": node["last_checked"],
                "uptime_seconds": node["uptime_seconds"]
            }, merge=True)
            count += 1
            if count >= self.batch_size:
                batch.commit()
                batch = self.db.batch()
                count = 0
        if count > 0:
            batch.commit()

    def get_node_status(self) -> Dict:
        nodes = {}
        for doc in self.db.collection("nodes").stream():
            nodes[doc.id] = doc.to_dict()
        return nodes

    def log_model_usage(self, model_name: str):
        batch = self.db.batch()
        ref = self.db.collection("model_usage").document(model_name)
        batch.set(ref, {
            "model_name": model_name,
            "request_count": firestore.Increment(1),
            "last_requested": time.time()
        }, merge=True)
        batch.commit()

    def get_model_usage(self) -> List[Dict]:
        usage = []
        for doc in self.db.collection("model_usage").stream():
            data = doc.to_dict()
            usage.append({
                "model_name": data["model_name"],
                "request_count": data.get("request_count", 0),
                "last_requested": data.get("last_requested", 0)
            })
        return usage

    def save_assignments(self, assignments: Dict):
        batch = self.db.batch()
        count = 0
        for assignment_type, assignment_list in assignments.items():
            for assignment in assignment_list:
                # Handle NodeAssignment objects
                models = [{"model": ma.model, "hot": ma.hot} for ma in assignment.models]
                ref = self.db.collection("assignments").document(assignment.node_id)
                batch.set(ref, {
                    "node_id": assignment.node_id,
                    "type": assignment_type,
                    "models": models,
                    "timestamp": time.time()
                }, merge=True)
                count += 1
                if count >= self.batch_size:
                    batch.commit()
                    batch = self.db.batch()
                    count = 0
        if count > 0:
            batch.commit()

    def get_assignments(self) -> Dict:
        assignments = {"gpu_assignments": [], "cpu_assignments": []}
        for doc in self.db.collection("assignments").stream():
            data = doc.to_dict()
            if data["type"] == "gpu":
                assignments["gpu_assignments"].append(data)
            else:
                assignments["cpu_assignments"].append(data)
        return assignments