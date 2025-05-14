import requests
from loguru import logger
import time
from typing import List,Dict,Any
from datetime import datetime, timedelta

SERVER_URL = "https://orchestrator-gekh.onrender.com"
API_PREFIX = "/api/v1"

def get_miner_list_with_resources() -> Dict[str, Dict[str, Any]]:
        """
        Fetch verified miners and their compute resources from the orchestrator API.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with miner IDs as keys and compute resources as values.
        """
        try:
            response = requests.get(f"{SERVER_URL}{API_PREFIX}/miners", timeout=100)
            response.raise_for_status()
            miners_data = response.json()
            return {
                miner["id"]: {
                    "compute_resources": miner["compute_resources"],
                    "status": "verified",
                    "last_checked": None,
                    "online": False,
                    "available": False
                }
                for miner in miners_data
                if miner["status"] == "verified"
            }
        except Exception as e:
            logger.error(f"Error fetching miner list with resources: {e}")
            return {}

def update_miner_status(miner_id: str, status: str, percentage: float, reason: str) -> str:
    updated_at =datetime.utcnow()
    url = f"https://orchestrator-gekh.onrender.com/api/v1/miners/{miner_id}/status"
    headers = {"Content-Type": "application/json"}
    payload = {"status": status,"Reason":reason, "updated_at":updated_at.isoformat() + "Z"}
    try:
        response = requests.patch(url, json=payload, headers=headers)
        response.raise_for_status()
        logger.info(f"Miner {miner_id} status updated to {status} with {percentage}%")
        return response.json().get("status", "unknown")
    except Exception as e:
        logger.error(f"Error updating miner {miner_id} status: {e}")
        return None

def get_containers_for_miner(miner_uid: str) -> list[str]:
    try:
        response = requests.get(f"https://orchestrator-gekh.onrender.com/api/v1/containers/miner/direct/{miner_uid}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching containers for miner {miner_uid}: {e}")
        return []

def get_miners_compute_resources() -> dict[str, dict]:
    """
    Retrieves compute resources for all miners.
    
    Returns:
        dict: A dictionary mapping miner IDs to their compute resources.
    """
    url = "https://orchestrator-gekh.onrender.com/api/v1/bittensor/miners/compute-resources"
    try:
        response = requests.get(url)
        response.raise_for_status()
        miners_data = response.json()
        return extract_miner_ids(miners_data)
    except Exception as e:
        logger.error(f"Error fetching miners compute resources: {e}")
        return {}

def get_miner_details(miner_id: str) -> dict:
    """
    Retrieves details for a specific miner by miner_id.
    
    Args:
        miner_id: The ID of the miner to retrieve details for.
    
    Returns:
        dict: A dictionary containing the miner's details, or an empty dict if the request fails.
    """
    url = f"https://orchestrator-gekh.onrender.com/api/v1/bittensor/miner/{miner_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        miner_data = response.json()
        logger.info(f"Retrieved details for miner {miner_id}")
        return miner_data
    except Exception as e:
        logger.error(f"Error fetching details for miner {miner_id}: {e}")
        return {}



def extract_miner_ids(data: List[dict]) -> List[str]:
    """
    Extract miner IDs from the 'multiple_miners_ips' list in the data.
    
    Args:
        data: List of dictionaries from get_miners_compute_resources().
    
    Returns:
        List of miner IDs (strings).
    """
    miner_ids = []
    
    try:
        # Validate input
        if not isinstance(data, list) or not data:
            logger.error("Data is not a non-empty list")
            return miner_ids
        
        # Access multiple_miners_ips from the first dict
        multiple_miners_ips = data[0].get("unique_miners_ips", [])
        if not isinstance(multiple_miners_ips, list):
            logger.error("multiple_miners_ips is not a list")
            return miner_ids
        
        # Extract keys from each dict in multiple_miners_ips
        for item in multiple_miners_ips:
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item: {item}")
                continue
            if len(item) != 1:
                logger.warning(f"Skipping dict with unexpected key count: {item}")
                continue
            miner_id = next(iter(item))  # Get the single key
            if isinstance(miner_id, str) and miner_id:
                miner_ids.append(miner_id)
            else:
                logger.warning(f"Skipping invalid miner ID: {miner_id}")
        
        logger.info(f"Extracted {len(miner_ids)} miner IDs")
        return miner_ids
    
    except Exception as e:
        logger.error(f"Error extracting miner IDs: {e}")
        return miner_ids