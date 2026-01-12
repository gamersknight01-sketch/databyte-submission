import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# =========================
# CONFIG
# =========================
DATA_PATH = r"C:\Users\megha\PycharmProjects\databyte\train.csv"
TEXT_COL = "text"
LABEL_COL = "target"   # 0 = Not Real, 1 = Real

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# BASIC TEXT CLEANING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"@\w+", "", text)             # remove mentions
    text = re.sub(r"#", "", text)                # keep hashtag words
    text = re.sub(r"[^\w\s]", "", text)          # punctuation
    return text

df["clean_text"] = df[TEXT_COL].apply(clean_text)

# =========================
# SPLIT BY CLASS
# =========================
real_text = " ".join(df[df[LABEL_COL] == 1]["clean_text"])
fake_text = " ".join(df[df[LABEL_COL] == 0]["clean_text"])

# =========================
# WORD CLOUD FUNCTION
# =========================
def plot_wordcloud(text, title):
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()

# =========================
# PLOT WORD CLOUDS
# =========================
plot_wordcloud(real_text, "Word Cloud - Real News")

plot_wordcloud(fake_text, "Word Cloud - Not Real (Fake) News")
