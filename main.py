# Step 1: Split the dataset into train and test
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
# Load the review_polarity_dataset stored as individual files grouped by target

samples = []
for target in ["pos", "neg"]:
    paths = Path(f"review_polarity/txt_sentoken/{target}").glob("*.txt")
    for pth in paths:
        with open(pth, "r") as f:
            samples.append({"content": f.read(), "target": target})

df = pd.DataFrame(samples)
df["is_positive"] = df["target"] == "pos"

X = df[["content"]].copy()
y = df["is_positive"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Use a popular embedding such as BERT to generate features
# You can use Hugging Face's Transformers library to load pre-trained BERT models
# and generate embeddings for your text data. Example:
from transformers import BertTokenizer, BertModel
import torch

device = torch.device("mps")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Tokenize and generate embeddings for your text data

def generate_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling over tokens
    return embeddings.detach().cpu().numpy()

from tqdm import tqdm

embeddings = [generate_bert_embeddings(text) for text in tqdm(X_train["content"])]

import numpy as np 
X_embedding = pd.DataFrame(np.vstack(embeddings))
X_embedding.index = X_train.index

# Step 3: Train a binary classifier.

# Example of generating BERT embeddings for a list of sentences
# bert_embeddings = [generate_bert_embeddings(sentence) for sentence in X_train]

# Step 3: Train a binary classifier
# Choose a classifier from scikit-learn (e.g., SVM, Logistic Regression) and train it
from sklearn.svm import SVC

classifier = SVC(kernel='linear')
classifier.fit(X_embedding, y_train)

# Step 4: Evaluate the dataset
# Use the trained classifier to make predictions on the test set and evaluate its performance
from sklearn.metrics import accuracy_score, classification_report

X_test_embedding = pd.DataFrame(np.vstack(
    [generate_bert_embeddings(text) for text in tqdm(X_test["content"])]
))
X_test_embedding.index = X_test.index

y_pred = classifier.predict(X_test_embedding)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

