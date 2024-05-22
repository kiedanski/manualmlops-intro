import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

class SentimentPredictor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('tokenizer')
        self.bert_model = BertModel.from_pretrained('bert_model')
        self.classifier = joblib.load('svm_classifier.joblib')

    def generate_bert_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.detach().cpu().numpy()

    def predict(self, text):
        embeddings = self.generate_bert_embeddings([text])
        prediction = self.classifier.predict(embeddings)
        return bool(prediction[0])

predictor = SentimentPredictor()

@app.route("/", methods=['GET'])
def home():
    html = "<h3>Model Serving</h3>"
    return html.format(format)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    instances = data['instances']
    predictions = [predictor.predict(instance['content']) for instance in instances]
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

