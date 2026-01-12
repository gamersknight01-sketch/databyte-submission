
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import SequenceClassifierOutput


DATA_PATH = r"C:\Users\megha\PycharmProjects\databyte\train.csv"
TEXT_COL = "text"
LABEL_COL = "target"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


df = pd.read_csv(DATA_PATH)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[TEXT_COL].tolist(),
    df[LABEL_COL].tolist(),
    test_size=0.2,
    stratify=df[LABEL_COL],
    random_state=42
)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
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


class TransformerWithFFN(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
            labels=None
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


def train_and_eval(model, train_ds, val_ds, freeze_encoder=False):

    if freeze_encoder and hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False

    model.to(DEVICE)

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

    model.eval()
    preds, labels = [], []

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE
    )

    with torch.no_grad():
        for batch in val_loader:
            labels.extend(batch["labels"].tolist())
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            preds.extend(
                torch.argmax(logits, dim=1).cpu().tolist()
            )

    return f1_score(labels, preds)


experiments = [
    ("bert-base-uncased", "default", False),
    ("bert-base-uncased", "ffn", False),
    ("roberta-base", "default", False),
    ("roberta-base", "ffn", False),
    ("roberta-base", "ffn", True)
]

results = {}

for model_name, head, freeze in experiments:
    print(f"\nRunning: {model_name} | {head} | freeze={freeze}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = NewsDataset(train_texts, train_labels, tokenizer)
    val_ds = NewsDataset(val_texts, val_labels, tokenizer)

    if head == "default":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
    else:
        model = TransformerWithFFN(model_name)

    f1 = train_and_eval(model, train_ds, val_ds, freeze)

    key = f"{model_name} | {head} | freeze={freeze}"
    results[key] = f1

    print(f"F1: {f1:.4f}")

best_model = max(results, key=results.get)

print("\nBEST CONFIGURATION ")
print("Model:", best_model)
print("Best F1:", round(results[best_model], 4))
