import asyncio
from typing import List, Dict
from utility.utils import load_data_from_db, get_available_nodes, flatten_models, assign_models_to_nodes, deploy_model, get_node_utilization, find_best_node_for_model
from db.database import FirestoreDB


class Scheduler:
    def __init__(self, db: FirestoreDB, interval: float):
        self.db = db
        self.interval = interval
        self.running = False

    async def scale_down(self, nodes: Dict, gpu_model_list: List[Dict], cpu_model_list: List[Dict]):
        for node_id, node in nodes.items():
            if not node["available"] or node["status"] != "online":
                continue
            metrics = get_node_utilization(node, node["compute_resources"][0]["resource_type"])
            utilization = metrics["gpu_utilization" if node["compute_resources"][0]["resource_type"] == "GPU" else "cpu_utilization"]
            if utilization < 0.2:
                doc = self.db.db.collection("assignments").document(node_id).get()
                if doc.exists:
                    assignment = doc.to_dict()
                    models = assignment["models"]
                    model_type = assignment["type"]
                    target_nodes = get_available_nodes(
                        {k: v for k, v in nodes.items() if k != node_id and v["available"]},
                        "GPU" if model_type == "gpu" else "CPU",
                        [], self.db.db
                    )
                    for model_assignment in models:
                        model = next(
                            (m for m in (gpu_model_list if model_type == "gpu" else cpu_model_list)
                            if m["name"] == model_assignment["model"]),
                            None
                        )
                        new_assignment = find_best_node_for_model(
                            target_nodes, model, is_gpu=model_type == "gpu", hot=model_assignment["hot"], db=self.db.db
                        )
                        if new_assignment:
                            new_node = next(n for n in nodes.values() if n["compute_resources"][0]["id"] == new_assignment["node_id"])
                            await asyncio.get_event_loop().run_in_executor(
                                None, deploy_model, new_node, model, model_assignment["hot"]
                            )
                            self.db.db.collection("assignments").document(new_assignment["node_id"]).set({
                                "node_id": new_assignment["node_id"],
                                "type": model_type,
                                "models": [{"model": model["name"], "hot": model_assignment["hot"]}],
                                "timestamp": self.db.db.server_timestamp()
                            }, merge=True)

                node["available"] = False
                self.db.db.collection("nodes").document(node_id).set({"available": False}, merge=True)
                self.db.db.collection("assignments").document(node_id).delete()

    async def scale_up(self, nodes: Dict, gpu_model_list: List[Dict], cpu_model_list: List[Dict]):
        active_nodes = [v for n, v in nodes.items() if v["available"] and v["status"] == "online"]
        if not active_nodes:
            return
        
        total_utilization = 0
        resource_type = active_nodes[0]["compute_resources"][0]["resource_type"]
        for node in active_nodes:
            metrics = get_node_utilization(node, node["compute_resources"][0]["resource_type"])
            total_utilization += metrics["gpu_utilization" if resource_type == "GPU" else "cpu_utilization"]
        
        avg_utilization = total_utilization / len(active_nodes)

        if avg_utilization > 0.8:
            reserve_nodes = [v for n, v in nodes.items() if not v["available"] and v["status"] == "online"]
            if reserve_nodes:
                node_to_activate = reserve_nodes[0]
                node_id = node_to_activate["compute_resources"][0]["id"]
                node_to_activate["available"] = True
                self.db.db.collection("nodes").document(node_id).set({"available": True}, merge=True)
                model_list = gpu_model_list if resource_type == "GPU" else cpu_model_list
                assignments = assign_models_to_nodes(
                    [node_to_activate], model_list, 3 if resource_type == "GPU" else 2, [], self.db.db
                )
                for assignment in assignments:
                    for model_assignment in assignment.models:
                        model = next((m for m in model_list if m["name"] == model_assignment.model), None)
                        await asyncio.get_event_loop().run_in_executor(
                            None, deploy_model, node_to_activate, model, model_assignment.hot
                        )
                    self.db.db.collection("assignments").document(assignment.node_id).set({
                        "node_id": assignment.node_id,
                        "type": "gpu" if resource_type == "GPU" else "cpu",
                        "models": [{"model": m.model, "hot": m.hot} for m in assignment.models],
                        "timestamp": self.db.db.server_timestamp()
                    }, merge=True)

    async def run(self):
        if self.running:
            return
        self.running = True
        try:
            while self.running:
                try:
                    gpu_models, cpu_models, nodes = load_data_from_db(self.db.db)
                    gpu_model_list, cpu_model_list = flatten_models(gpu_models, cpu_models)
                    
                    await self.scale_down(nodes, gpu_model_list, cpu_model_list)
                    await self.scale_up(nodes, gpu_model_list, cpu_model_list)
                    
                    self.db.update_node_status(nodes)
                    gpu_nodes = get_available_nodes(nodes, "GPU", [], self.db.db)
                    cpu_nodes = get_available_nodes(nodes, "CPU", [], self.db.db)
                    gpu_assignments = assign_models_to_nodes(gpu_nodes, gpu_model_list, 3, [], self.db.db)
                    cpu_assignments = assign_models_to_nodes(cpu_nodes, cpu_model_list, 2, [], self.db.db)
                    
                    for assignment in gpu_assignments + cpu_assignments:
                        node = next((n for n in nodes.values() if n["compute_resources"][0]["id"] == assignment.node_id), None)
                        for model_assignment in assignment.models:
                            model = next((m for m in (gpu_model_list + cpu_model_list) if m["name"] == model_assignment.model), None)
                            await asyncio.get_event_loop().run_in_executor(
                                None, deploy_model, node, model, model_assignment.hot
                            )
                    
                    self.db.save_assignments({"gpu": gpu_assignments, "cpu": cpu_assignments})
                    
                except Exception as e:
                    print(f"Scheduler error: {str(e)}")
                
                await asyncio.sleep(self.interval)
        finally:
            self.running = False