import pytest
import requests

API_URL = "http://127.0.0.1:8000/predict"

@pytest.mark.parametrize("text", [
    "I love this product!",
    "It's okay, not the best.",
    "I hate waiting in long lines."
])
def test_sentiment_api(text):
    response = requests.post(API_URL, json={"text": text})
    assert response.status_code == 200, f"Request failed for input: {text}"
    
    data = response.json()
    assert "label" in data, "Missing 'label' in response"
    assert "score" in data, "Missing 'score' in response"
    assert data["label"] in ["positive", "neutral", "negative", "POSITIVE", "NEUTRAL", "NEGATIVE"], "Unexpected sentiment label"
    assert isinstance(data["score"], float), "Score is not a float"
