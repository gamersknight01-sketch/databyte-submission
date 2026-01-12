import pandas as pd
import re

# Load dataset
df = pd.read_csv(r"C:\Users\megha\PycharmProjects\databyte\train.csv")

# Adjust column names if needed
TEXT_COL = "text"    # change if your column name is different

# =========================
# Noise detection functions
# =========================

def has_http(text):
    return int(bool(re.search(r"http[s]?://|www\.", str(text))))

def has_mention(text):
    return int(bool(re.search(r"@\w+", str(text))))

def has_non_ascii(text):
    return int(any(ord(char) > 127 for char in str(text)))

# =========================
# Represent noises as columns
# =========================

df["noise_http"] = df[TEXT_COL].apply(has_http)
df["noise_mention"] = df[TEXT_COL].apply(has_mention)
df["noise_non_ascii"] = df[TEXT_COL].apply(has_non_ascii)

# Quick sanity check
print(df[[TEXT_COL, "noise_http", "noise_mention", "noise_non_ascii"]].head())
