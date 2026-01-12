
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

DATA_PATH = r"C:\Users\megha\PycharmProjects\databyte\train.csv"

df = pd.read_csv(DATA_PATH)

TEXT_COL = "text"
LABEL_COL = "target"

X_text = df[TEXT_COL]
y = df[LABEL_COL]

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(
        max_iter=2000,
        solver="liblinear"
    ))
])

param_grid = {
    # TF-IDF parameters
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__min_df": [2, 5],
    "tfidf__max_df": [0.8, 0.9],
    "tfidf__sublinear_tf": [True],

    # Logistic Regression parameters
    "clf__C": [0.1, 0.5, 1, 2],
    "clf__class_weight": [
        {0: 1, 1: 1},
        {0: 1, 1: 1.3},
        {0: 1, 1: 1.6}
    ]
}
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train_text, y_train)

best_model = grid.best_estimator_

print("\nBest CV F1:", grid.best_score_)
print("\nBest Parameters:")
for k, v in grid.best_params_.items():
    print(f"{k}: {v}")

y_val_pred = best_model.predict(X_val_text)
print("\nClassification Report (threshold=0.5):\n")
print(classification_report(y_val, y_val_pred))

probs = best_model.predict_proba(X_val_text)[:, 1]

thresholds = np.arange(0.2, 0.8, 0.01)
f1_scores = []

for t in thresholds:
    preds = (probs >= t).astype(int)
    f1_scores.append(f1_score(y_val, preds))

best_threshold = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print("\nBest Threshold:", round(best_threshold, 2))
print("Best F1 After Threshold Tuning:", round(best_f1, 4))

final_preds = (probs >= best_threshold).astype(int)

print("\nFinal Classification Report (Optimized Threshold):\n")
print(classification_report(y_val, final_preds))
