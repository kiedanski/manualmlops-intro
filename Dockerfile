FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest

RUN pip install transformers torch scikit-learn 
RUN pip install flask joblib

# Copy model and predictor script
COPY tokenizer /app/tokenizer
COPY bert_model /app/bert_model
COPY svm_classifier.joblib /app/
COPY predictor.py /app/

# Set the working directory
WORKDIR /app

# Expose the port
EXPOSE 8080

# Define the entrypoint
ENTRYPOINT [ "python" ]

CMD ["predictor.py"]
