import requests

# Esempio di test locale (assumendo che FastAPI sia in esecuzione su http://127.0.0.1:8000)
API_URL = "http://127.0.0.1:8000/reach"

sample_texts = [
    "Dummy text for testing.",
]

for text in sample_texts:
    response = requests.post(API_URL, json={"text": text})
    print(f"Input: {text}")
    print(f"Response code: {response.status_code}")
    print(f"Response: {response.json()}")
    print("---")
