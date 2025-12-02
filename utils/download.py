import os
from azure.storage.blob import BlobServiceClient


def download_model():
    conn_str = "DefaultEndpointsProtocol=https;AccountName=synthauditor;AccountKey=tVbibIqXWK4VYrRzZ0GHgsSJC2AQ8RLBaaDotQG/TLvRLf+WF74xLmPkgD6MwPNb6i1L8FFdOHf3+AStzczhwg==;EndpointSuffix=core.windows.net
    container_name = "model"
    blob_name = "yaari-synth-auditor-model-v1.pth"
    local_path = "./model/yaari-synth-auditor-model-v1.pth"
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
