from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os
import csv
from pathlib import Path
from datetime import datetime, timezone
from fastapi.responses import JSONResponse

app = FastAPI(title="Sentiment Analysis API")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_DIR = "./model"
MODEL_LOG_PATH = "model/answers.csv"  # absolute path inside container

def load_or_download_model():
    """
    Load the model and tokenizer from local directory if available, otherwise download from Hugging Face.
    """
    if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR):
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            print("Loaded model from local folder.")
        except Exception as e:
            print(f"Failed to load local model: {e}")
            tokenizer, model = download_and_save_model()
    else:
        tokenizer, model = download_and_save_model()

    return tokenizer, model

def download_and_save_model():
    """"
    Download the model and tokenizer from Hugging Face and save them locally.
    """
    print("Downloading model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    os.makedirs(MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print("Model saved to local folder.")
    return tokenizer, model

# Load pipeline
tokenizer, model = load_or_download_model()
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# API schemas
class TextInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    """
    Response model for sentiment analysis output.
    """
    label: str
    score: float

@app.post("/predict", response_model=SentimentOutput)
def predict_sentiment(input: TextInput):
    """
    Predict sentiment of the input text and log the prediction.
    """
    result = classifier(input.text)[0]
    label = result['label']
    score = result['score']

    # Log prediction
    log_prediction(input.text, label, score)

    return SentimentOutput(label=label, score=score)


def log_prediction(text: str, label: str, score: float, path: str = MODEL_LOG_PATH):
    """
    Log the prediction to a CSV file.
    Useful for tracking model performance and debugging.
    """
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_path.exists()
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "text", "predicted_label", "score"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), text, label, score])


@app.get("/answers")
def get_logged_predictions():
    """
    Retrieve logged predictions from the CSV file.
    Useful for monitoring (Grafana).
    """
    log_path = Path(MODEL_LOG_PATH)
    if not log_path.exists():
        return []

    with open(log_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "timestamp": row["timestamp"],
                "text": row["text"],
                "predicted_label": row["predicted_label"],
                "score": float(row["score"])
            })
        return rows

@app.get("/test-write")
def test_write_csv():
    from datetime import datetime, timezone
    path = Path(MODEL_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "text", "predicted_label", "score"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), "test", "POSITIVE", 0.99])
    return {"status": "written"}

@app.post("/reach", response_model=str)
def test_endpoint(input: TextInput):
    """
    Test endpoint to check if the API is running.
    """
    return "Reached!"