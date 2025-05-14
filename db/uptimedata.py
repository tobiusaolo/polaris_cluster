import requests
import datetime
import json
from typing import Dict, Any, Optional
import logging
from datetime import timezone
import tenacity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configuration
SERVER_URL = "https://orchestrator-gekh.onrender.com"
API_PREFIX = "/api/v1"

@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def check_node_uptime(miner_id: str) -> Dict[str, Any]:
    """
    Check the uptime of a miner by querying the state API.
    
    Args:
        miner_id (str): The ID of the miner to check.
    
    Returns:
        Dict[str, Any]: Dictionary containing uptime and status information.
    """
    try:
        state_url = f"{SERVER_URL}{API_PREFIX}/miners/{miner_id}/state"
        state_response = requests.get(state_url, timeout=10)
        state_response.raise_for_status()
        state_data = state_response.json()

        current_status = state_data.get("current_status", "offline")
        current_metrics = state_data.get("current_metrics", {})
        system_info = current_metrics.get("system_info", {})
        current_uptime = system_info.get("uptime", 0)
        last_heartbeat = state_data.get("last_heartbeat", None)

        # Parse last heartbeat time
        time_since_heartbeat = None
        if last_heartbeat:
            try:
                last_heartbeat_time = datetime.datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                time_since_heartbeat = (datetime.datetime.now(timezone.utc) - last_heartbeat_time).total_seconds()
            except Exception as e:
                logger.error(f"Error parsing heartbeat timestamp for {miner_id}: {e}")

        return {
            "miner_id": miner_id,
            "status": current_status,
            "uptime_seconds": current_uptime,
            "time_since_heartbeat": time_since_heartbeat,
            "last_checked": datetime.datetime.now(timezone.utc).isoformat()
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching state for {miner_id}: {e}")
        return {
            "miner_id": miner_id,
            "status": "offline",
            "uptime_seconds": 0,
            "time_since_heartbeat": None,
            "last_checked": datetime.datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }
