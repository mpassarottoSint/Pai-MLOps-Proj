import requests

# Esempio di test locale (assumendo che FastAPI sia in esecuzione su http://127.0.0.1:8000)
API_URL = "http://127.0.0.1:8000/answers"

response = requests.get(API_URL)
print(f"Response code: {response.status_code}")
print(f"Response: {response.json()}")
print("---")
