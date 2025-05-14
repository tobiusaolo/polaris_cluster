import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from loguru import logger as loguru_logger
from db.api_utils import get_miner_list_with_resources
from db.pogs import execute_ssh_tasks
from db.uptimedata import check_node_uptime


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
loguru_logger.configure(handlers=[{"sink": os.sys.stderr, "level": "INFO"}])

# API configuration
SERVER_URL = "https://orchestrator-gekh.onrender.com"
API_PREFIX = "/api/v1"

class NodeManager:
    def __init__(self, node_db_path: str = "nodes.json"):
        """
        Initialize the Node Manager with a node database path for persistent storage.
        
        Args:
            node_db_path (str): Path to store node status data.
        """
        self.node_db_path = node_db_path
        self.nodes: Dict[str, Dict[str, Any]] = {}  # In-memory node registry
        self.load_nodes()
        logger.info("Node Manager initialized")

    def load_nodes(self):
        """Load node data from the database file if it exists."""
        if os.path.exists(self.node_db_path):
            try:
                with open(self.node_db_path, 'r') as f:
                    self.nodes = json.load(f)
                logger.info(f"Loaded node data from {self.node_db_path}")
            except Exception as e:
                logger.error(f"Error loading node data: {e}")
                self.nodes = {}

    def save_nodes(self):
        """Save node data to the database file."""
        try:
            with open(self.node_db_path, 'w') as f:
                json.dump(self.nodes, f, indent=2)
            logger.info(f"Saved node data to {self.node_db_path}")
        except Exception as e:
            logger.error(f"Error saving node data: {e}")


    def validate_node(self, miner_id: str) -> bool:
        """
        Validate a node's online status by checking uptime and SSH availability.
        
        Args:
            miner_id (str): The ID of the miner to validate.
        
        Returns:
            bool: True if the node is online and available, False otherwise.
        """
        # Check uptime via state API
        uptime_info = check_node_uptime(miner_id)
        if uptime_info["status"] != "online":
            logger.info(f"Node {miner_id} is not online (status: {uptime_info['status']})")
            return False

        # Validate via SSH tasks
        ssh_result = execute_ssh_tasks(miner_id)
        if not ssh_result["is_online"]:
            logger.info(f"Node {miner_id} is not considered online due to non-empty task_results")
            return False

        # Node is online and has no task_results
        logger.info(f"Node {miner_id} validated as online and available")
        return True

    def update_node_status(self):
        """
        Update the status of all nodes by fetching miners and validating their uptime/SSH.
        """
        # Fetch verified miners
        miners = get_miner_list_with_resources()
        logger.info(f"Fetched {len(miners)} verified miners")

        for miner_id, miner_data in miners.items():
            # Validate node
            is_online = self.validate_node(miner_id)
            is_available = is_online  # For now, online nodes are considered available

            # Update node registry
            self.nodes[miner_id] = {
                "compute_resources": miner_data["compute_resources"],
                "status": "online" if is_online else "offline",
                "available": is_available,
                "last_checked": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": check_node_uptime(miner_id).get("uptime_seconds", 0)
            }

        # Save updated node data
        self.save_nodes()
        logger.info("Node status update completed")

    def get_online_nodes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of online and available nodes.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of online nodes with their details.
        """
        return {k: v for k, v in self.nodes.items() if v["status"] == "online" and v["available"]}

    def select_node(self, required_vram: Optional[int] = None) -> Optional[str]:
        """
        Select an online and available node, optionally filtering by VRAM requirement.
        
        Args:
            required_vram (Optional[int]): Minimum VRAM required in GB.
        
        Returns:
            Optional[str]: Miner ID of the selected node, or None if no suitable node is found.
        """
        online_nodes = self.get_online_nodes()
        if not online_nodes:
            logger.warning("No online nodes available")
            return None

        # Filter nodes by VRAM if specified
        suitable_nodes = online_nodes
        if required_vram:
            suitable_nodes = {
                k: v for k, v in online_nodes.items()
                if v["compute_resources"].get("gpu_vram_gb", 0) >= required_vram
            }

        if not suitable_nodes:
            logger.warning(f"No nodes with sufficient VRAM ({required_vram} GB)")
            return None

        # Select node with highest uptime (or use another metric, e.g., least loaded)
        selected_node = max(suitable_nodes.items(), key=lambda x: x[1]["uptime_seconds"])[0]
        logger.info(f"Selected node {selected_node} with uptime {suitable_nodes[selected_node]['uptime_seconds']} seconds")
        return selected_node

node_manager = NodeManager(node_db_path="nodes.json")

# Update node status
node_manager.update_node_status()

# Get online nodes
online_nodes = node_manager.get_online_nodes()
print("Online nodes:", online_nodes)

# Select a node with at least 16 GB VRAM
selected_node = node_manager.select_node(required_vram=16)
print("Selected node:", selected_node)