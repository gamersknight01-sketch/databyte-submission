# =========================================================
# IMPORTS
# =========================================================
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = r"C:\Users\megha\PycharmProjects\databyte\train.csv"

TEXT_COL = "text"
LABEL_COL = "target"

MODEL_NAME = "roberta-base"     # 🔒 LOCKED BEST MODEL
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(DATA_PATH)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[TEXT_COL].tolist(),
    df[LABEL_COL].tolist(),
    test_size=0.2,
    stratify=df[LABEL_COL],
    random_state=42
)

# =========================================================
# TOKENIZER
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================================================
# DATASET
# =========================================================
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_ds = NewsDataset(train_texts, train_labels)
val_ds   = NewsDataset(val_texts, val_labels)

# =========================================================
# MODEL (DEFAULT HEAD)
# =========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(DEVICE)

# =========================================================
# OPTIMIZER & SCHEDULER
# =========================================================
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True
)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

scaler = GradScaler()

# =========================================================
# TRAIN
# =========================================================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f}")

# =========================================================
# EVALUATION
# =========================================================
model.eval()
preds, labels = [], []

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

with torch.no_grad():
    for batch in val_loader:
        labels.extend(batch["labels"].tolist())
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        preds.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())

f1 = f1_score(labels, preds)

print("\nFinal Validation F1:", round(f1, 4))
print("\nClassification Report:\n")
print(classification_report(labels, preds))
