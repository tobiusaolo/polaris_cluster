import json
import logging
import requests
from typing import Dict, Any
import tenacity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("remote_access")

SERVER_URL = "https://orchestrator-gekh.onrender.com"
API_PREFIX = "/api/v1"

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
def execute_ssh_tasks(miner_id: str) -> Dict[str, Any]:
    """
    Execute SSH tasks for a miner to validate its availability.
    
    Args:
        miner_id (str): The ID of the miner to execute tasks for.
    
    Returns:
        Dict[str, Any]: Dictionary containing task execution status and results.
    """
    logger.info(f"Executing SSH tasks for miner {miner_id}")
    url = f"{SERVER_URL}{API_PREFIX}/miners/{miner_id}/perform-tasks"

    try:
        response = requests.get(url, timeout=30)
        logger.info(f"Response status for {miner_id}: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            logger.debug(f"Full server response for {miner_id}: {json.dumps(result, indent=2)}")

            if result.get("status") != "success":
                logger.error(f"Server error for {miner_id}: {result.get('message', 'Unknown error')}")
                return {
                    "status": "error",
                    "message": result.get("message", "Server reported failure"),
                    "task_results": {},
                    "is_online": False
                }

            task_results = result.get("task_results", result.get("specifications", {}))
            if not isinstance(task_results, dict):
                logger.error(f"Invalid task_results format for {miner_id}: {task_results}")
                return {
                    "status": "error",
                    "message": "Invalid task_results format",
                    "task_results": {},
                    "is_online": False
                }

            logger.info(f"SSH tasks executed for {miner_id}, task_results: {task_results}")
            # Consider miner online if task_results has valid specs and no errors
            is_online = bool(task_results.get("cpu_specs") or task_results.get("ram")) and not task_results.get("error")
            return {
                "status": "success",
                "message": "SSH tasks executed successfully",
                "task_results": task_results,
                "is_online": is_online
            }
        elif response.status_code == 404:
            logger.error(f"Miner {miner_id} not found")
            return {
                "status": "error",
                "message": "Miner not found",
                "task_results": {},
                "is_online": False
            }
        else:
            logger.error(f"Unexpected status code for {miner_id}: {response.status_code}")
            return {
                "status": "error",
                "message": f"Server returned status code {response.status_code}",
                "task_results": {},
                "is_online": False
            }
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out for {miner_id}")
        return {
            "status": "error",
            "message": "Request timed out",
            "task_results": {},
            "is_online": False
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {miner_id}: {e}")
        return {
            "status": "error",
            "message": f"Request error: {str(e)}",
            "task_results": {},
            "is_online": False
        }
    except Exception as e:
        logger.error(f"Unexpected error executing SSH tasks for {miner_id}: {e}")
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "task_results": {},
            "is_online": False
        }