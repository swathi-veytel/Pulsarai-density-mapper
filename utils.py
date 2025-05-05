# utils.py
import os
import pandas as pd
from google.cloud import storage

GCS_BUCKET_NAME = "veytel-cloud-store"
GCS_FOLDER_PATH = "density_mapper"

def authenticate_gcs(service_account_info):
    return storage.Client.from_service_account_info(service_account_info)

def download_csv_from_gcs(service_account_info, filename):
    client = authenticate_gcs(service_account_info)
    bucket = client.get_bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(os.path.join(GCS_FOLDER_PATH, filename))
    return blob.download_as_text()

def read_user_csv(service_account_info, user):
    filename = f"density_mapper_v8_{user}.csv"
    try:
        csv_content = download_csv_from_gcs(service_account_info, filename)
        rows = csv_content.strip().split("\n")
        return [row.split(",") for row in rows]
    except Exception:
        return []

USERS = ["GK", "Siddique", "Nameer", "Taaha", "Konstantine","Vijayakumar","Swathi", "Ellen", "Cathy", "Robin", "Anrey", "Clara",  "Song", "Kevin",
                                            "Claire", "Rachel", "Mike", "Paul", "Test_1", "Test_2", "Test_3", "Test_4", "Test_5", "Test_6", "Test_7", "Test_8", "Test_9", "Test_10", "Expert_Annotator_1", "Expert_Annotator_2", "Expert_Annotator_3", "Expert_Annotator_4", "Expert_Annotator_5", "Expert_Annotator_6"]
