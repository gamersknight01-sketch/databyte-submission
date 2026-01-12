
import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

DATA_PATH = r"C:\Users\megha\PycharmProjects\databyte\train.csv"

df = pd.read_csv(DATA_PATH)

TEXT_COL = "text"
LABEL_COL = "target"

STOPWORDS = set(stopwords.words("english"))

def preprocess_text(text):
    text = str(text).lower()                              # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)            # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    tokens = text.split()                                 # tokenization
    tokens = [t for t in tokens if t not in STOPWORDS]    # remove stopwords
    return " ".join(tokens)

df["clean_text"] = df[TEXT_COL].apply(preprocess_text)

X_train, X_val, y_train, y_val = train_test_split(
    df["clean_text"],
    df[LABEL_COL],
    test_size=0.2,
    stratify=df[LABEL_COL],
    random_state=42
)


tfidf = TfidfVectorizer(
    ngram_range=(1, 1),        # unigrams only (classic)
    min_df=2,
    max_df=0.9,
    sublinear_tf=True          # ⭐ unique but valid
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf   = tfidf.transform(X_val)

model = LogisticRegression(
    max_iter=2000,
    solver="liblinear"
)

model.fit(X_train_tfidf, y_train)

y_val_pred = model.predict(X_val_tfidf)

f1 = f1_score(y_val, y_val_pred)

print("Baseline F1-score:", round(f1, 4))
print("\nClassification Report:\n")
print(classification_report(y_val, y_val_pred))
