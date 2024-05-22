import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch
import numpy as np
from tqdm import tqdm
from google.cloud import storage
import joblib

GCP_BUCKET = 'manualmlops-review-classiifcation'
RAW_DATA_PATH = 'review_polarity.csv'
OUTPUT_DATASET_PATH ='review_polarity.csv'

def download_data(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def upload_model(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Model {source_file_name} uploaded to {destination_blob_name}.")

def upload_directory(bucket_name, source_directory, destination_blob_name):
    """Uploads a directory to the bucket."""
    subprocess.check_call(['gsutil', '-m', 'cp', '-r', source_directory, f'gs://{bucket_name}/{destination_blob_name}'])

def generate_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling over tokens
    return embeddings.detach().cpu().numpy()

def train_model():
    # Replace with your GCP bucket and file names
    destination_file_name = '/tmp/preprocessed_data.csv'
    
    download_data(GCP_BUCKET, RAW_DATA_PATH, destination_file_name)
    
    df = pd.read_csv(destination_file_name).sample(50)
    df["is_positive"] = df["target"] == "pos"
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    embeddings = [generate_bert_embeddings(text, tokenizer, model) for text in tqdm(df["content"])]
    df_embeddings = pd.DataFrame(np.vstack(embeddings))


    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df_embeddings, df["is_positive"], test_size=0.2, random_state=42
    )


    # Training
    classifier = SVC(kernel="linear", C=0.01)
    classifier.fit(X_train, y_train)

    # Evaluation
    y_pred = classifier.predict(X_test)


    print(classification_report(y_test, y_pred))


    # Saving the models
    tokenizer.save_pretrained("tokenizer")
    model.save_pretrained("bert")
    joblib.dump(classifier, "classifier.joblib")

    upload_directory(GCP_BUCKET, "tokenizer", "tokenizer")
    upload_directory(GCP_BUCKET, "bert", "bert")
    upload_model(GCP_BUCKET, "classifier.joblib", "svm_classifier.joblib")

if __name__ == "__main__":
    train_model()

