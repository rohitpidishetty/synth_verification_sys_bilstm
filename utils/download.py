import os
from azure.storage.blob import BlobServiceClient


def download_model():
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("MODEL_CONTAINER_NAME")
    blob_name = os.getenv("MODEL_BLOB_NAME")
    local_path = os.getenv("MODEL_LOCAL_PATH")

    if not os.path.exists(local_path):
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as file:
            blob_data = blob_client.download_blob()
            blob_data.readinto(file)
        print(f"Downloaded model to {local_path}")
    else:
        print(f"Model already exists at {local_path}")
