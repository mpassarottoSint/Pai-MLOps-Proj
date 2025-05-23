import requests

# Esempio di test locale (assumendo che FastAPI sia in esecuzione su http://127.0.0.1:8000)
API_URL = "http://127.0.0.1:8000/predict"

sample_texts = [
    "I love this product!",
    "It's okay, not the best.",
    "I hate waiting in long lines.",
    "Music is boring.",
    "I don't have time.",
    "Today it's a meh.",
    "Tomoworrow is a great day.",
    "I am not sure about this.",
    "It's a sunny day.",
    "I am happy with the results.",
    "I am sad about the news.",
    "I am excited for the weekend.",
    "I am angry about the situation.",
    "I am confused about the instructions.",
    "I am surprised by the outcome.",
]

for text in sample_texts:
    response = requests.post(API_URL, json={"text": text})
    print(f"Input: {text}")
    print(f"Response: {response.json()}")
    print("---")
