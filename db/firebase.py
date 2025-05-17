# app/db/firebase.py
import firebase_admin
from firebase_admin import credentials, firestore
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase app
def init_firebase():
    try:
        # Check if default app exists
        firebase_admin.get_app(name="[DEFAULT]")
        logger.info("Default Firebase app already initialized")
    except ValueError:
        # Initialize app if it doesn't exist
        try:
            service_account_info = json.load(open("db/creds.json"))
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase app initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            raise
    return firestore.client()

# Initialize Firebase client
db = init_firebase()