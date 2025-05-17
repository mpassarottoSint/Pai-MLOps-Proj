from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

# Carica modello
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Carica dataset
dataset = load_dataset("sentiment140", split="train[:1000]")  # Subset per rapidità

# Mappa etichette numeriche in label stringa
def map_label(score):
    return {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}.get(score, "NEUTRAL")

true_labels = []
pred_labels = []

for entry in dataset:
    text = entry['text']
    true_label = map_label(entry['sentiment'])
    pred = classifier(text)[0]['label']
    true_labels.append(true_label)
    pred_labels.append(pred)

print(classification_report(true_labels, pred_labels, digits=4))
