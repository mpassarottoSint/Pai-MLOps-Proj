from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
import json
from pathlib import Path
import torch

import transformers
print(f"Transformer version: {transformers.__version__}")

# Model
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Dataset
dataset = load_dataset("sentiment140", trust_remote_code=True)
label_map = {0: 0, 2: 1, 4: 2}  # map original labels to 0, 1, 2

def preprocess(example):
    encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    encoding["label"] = [label_map[s] for s in example["sentiment"]]
    return encoding

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Metric
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# Training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].shuffle(seed=42).select(range(5000)),  # subset
    eval_dataset=dataset["train"].shuffle(seed=42).select(range(500, 1000)),
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate on validation set and save metrics
metrics = trainer.evaluate()
metrics_path = Path("./model/metrics.json")
metrics_path.parent.mkdir(parents=True, exist_ok=True)

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

# Optional: log to plain text
with open("./model/train.log", "w") as f:
    f.write("Training complete.\n")
    f.write("Final Evaluation Metrics:\n")
    for k, v in metrics.items():
        f.write(f"{k}: {v:.4f}\n")

# Save the model
model.save_pretrained("./model/")
tokenizer.save_pretrained("./model/")
