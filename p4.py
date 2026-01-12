import pandas as pd
import re
import matplotlib.pyplot as plt

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
# NOISE PATTERNS
# =========================
patterns = {
    "HTTP": r"http[s]?://\S+|www\.\S+",
    "MENTION": r"@\w+",
    "HASHTAG": r"#\w+",
    "NON_ASCII": r"[^\x00-\x7F]"
}

# =========================
# CREATE NOISE FLAGS
# =========================
for name, pattern in patterns.items():
    df[name] = df[TEXT_COL].apply(
        lambda x: bool(re.search(pattern, str(x)))
    )

# =========================
# PLOTTING FUNCTION
# =========================
def plot_noise_distribution(noise_col, title):
    plt.figure()
    summary = (
        df.groupby([noise_col, LABEL_COL])
        .size()
        .unstack(fill_value=0)
    )

    # Rename target labels for readability
    summary.columns = ["Not Real", "Real"]

    summary.plot(
        kind="bar",
        stacked=True,
        figsize=(6, 4)
    )

    plt.title(title)
    plt.xlabel(f"{noise_col} Present?")
    plt.ylabel("Number of Samples")
    plt.legend(title="Class")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# =========================
# VISUALIZATIONS
# =========================
plot_noise_distribution(
    "HTTP",
    "Real vs Not Real: With and Without HTTP Links"
)

plot_noise_distribution(
    "MENTION",
    "Real vs Not Real: With and Without User Mentions"
)

plot_noise_distribution(
    "HASHTAG",
    "Real vs Not Real: With and Without Hashtags"
)

plot_noise_distribution(
    "NON_ASCII",
    "Real vs Not Real: With and Without Non-ASCII Characters"
)
