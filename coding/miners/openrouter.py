import os
import json
import requests
import bittensor as bt
from uuid import uuid4
from coding.protocol import ProvisionKeySynapse

BASE_URL = "https://openrouter.ai/api/v1/keys"

def miner_process(self, synapse: ProvisionKeySynapse) -> ProvisionKeySynapse:
    """
    Process requests for OpenRouter API key provisioning
    """
    bt.logging.info(f"Processing OpenRouter provisioning request: {synapse.action}")

    # Get the provisioning key from environment
    provisioning_key = os.environ.get("PROVISIONING_API_KEY")

    if not provisioning_key:
        bt.logging.error("PROVISIONING_API_KEY not found in environment")
        return synapse

    # Handle key creation
    if synapse.action == "create":
        try:
            # Create a temporary key using the provisioning key
            response = requests.post(
                "https://openrouter.ai/api/v1/auth/keys",
                headers={"Authorization": f"Bearer {provisioning_key}"},
                json={"name": "Temporary validator key"}
            )

            if response.status_code == 200:
                key_data = response.json()
                synapse.api_key = key_data.get("key")
                synapse.key_hash = key_data.get("id")
                bt.logging.info(f"Successfully created temporary API key with hash: {synapse.key_hash}")
            else:
                bt.logging.error(f"Failed to create API key: {response.text}")

        except Exception as e:
            bt.logging.error(f"Error creating OpenRouter API key: {e}")

    # Handle key deletion
    elif synapse.action == "delete" and synapse.key_hash:
        try:
            # Delete the temporary key
            response = requests.delete(
                f"https://openrouter.ai/api/v1/auth/keys/{synapse.key_hash}",
                headers={"Authorization": f"Bearer {provisioning_key}"}
            )

            if response.status_code == 200:
                bt.logging.info(f"Successfully deleted key with hash: {synapse.key_hash}")
            else:
                bt.logging.error(f"Failed to delete key: {response.text}")

        except Exception as e:
            bt.logging.error(f"Error deleting OpenRouter API key: {e}")

    return synapse
