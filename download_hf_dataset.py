
from datasets import load_dataset
import pandas as pd
import os


OUT_DIR    = r"C:\Users\shaur\PhishingDetectorProject\dataset"
RAW_FILE   = os.path.join(OUT_DIR, "hf_phishing_raw.csv")
CLEAN_FILE = os.path.join(OUT_DIR, "hf_phishing_clean.csv")
os.makedirs(OUT_DIR, exist_ok=True)


ds = load_dataset("zefang-liu/phishing-email-dataset")  
df_raw = ds["train"].to_pandas()


df_raw.to_csv(RAW_FILE, index=False, encoding="utf-8")
print(f" Raw HF dataset saved to {RAW_FILE}")
print("  Columns detected:", df_raw.columns.tolist())


keep = ["Email Text", "Email Type"]
existing = [c for c in keep if c in df_raw.columns]
df = df_raw[existing].copy()


df = df.rename(columns={
    "Email Text": "email_text",
    "Email Type": "label"
})


if df["label"].dtype == object:
    df["label"] = df["label"].map({
        "Safe Email": 0,
        "Phishing Email": 1
    })

df.to_csv(CLEAN_FILE, index=False, encoding="utf-8")
print(f" Clean HF dataset saved to {CLEAN_FILE}")
print(f"   Total rows: {len(df)}")
print(f"   Label distribution:\n{df['label'].value_counts()}")
