from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

app = FastAPI(title="Sentiment Analysis API")

# Caricamento modello
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

class TextInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=SentimentOutput)
def predict_sentiment(input: TextInput):
    result = classifier(input.text)[0]
    return SentimentOutput(label=result['label'], score=result['score'])
