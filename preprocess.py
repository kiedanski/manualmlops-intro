import pandas as pd
from google.cloud import storage
from pathlib import Path
import os

GCP_BUCKET = 'manualmlops-review-classiifcation'
RAW_DATA_PATH = 'review_polarity.zip'
OUTPUT_DATASET_PATH ='review_polarity.csv'


def download_data(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client(project="mlops-devrel")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def upload_data(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(project="mlops-devrel")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def preprocess_data():
    # Replace with your GCP bucket and file names
    destination_file_name = '/tmp/review_polarity.zip'
    
    download_data(GCP_BUCKET, RAW_DATA_PATH, destination_file_name)
    
    # Unzip and preprocess
    import zipfile
    with zipfile.ZipFile(destination_file_name, 'r') as zip_ref:
        zip_ref.extractall('/tmp/')
    
    samples = []
    for target in ["pos", "neg"]:
        paths = Path(f"/tmp/review_polarity/txt_sentoken/{target}").glob("*.txt")
        for pth in paths:
            with open(pth, "r") as f:
                samples.append({"content": f.read(), "target": target})


    df = pd.DataFrame(samples)
    df["is_positive"] = df["target"] == "pos"
    
    preprocessed_data_path = '/tmp/preprocessed_data.csv'
    df.to_csv(preprocessed_data_path, index=False)
    print("Data preprocessed and saved to /tmp/preprocessed_data.csv")
    
    # Upload preprocessed data to GCP
    upload_data(GCP_BUCKET, preprocessed_data_path, OUTPUT_DATASET_PATH)

if __name__ == "__main__":
    preprocess_data()

