from datasets import load_dataset  # to download the phishing-email-dataset
import pandas as pd               # for data handling (DataFrames, CSV)
import os                         # for file path operations

# 1. Set up paths for raw and cleaned files
OUT_DIR    = r"C:\Users\shaur\PhishingDetectorProject\dataset"
RAW_FILE   = os.path.join(OUT_DIR, "hf_phishing_raw.csv")
CLEAN_FILE = os.path.join(OUT_DIR, "hf_phishing_clean.csv")
os.makedirs(OUT_DIR, exist_ok=True)  # make sure dataset folder exists

# 2. Download and load the Hugging Face dataset
ds = load_dataset("zefang-liu/phishing-email-dataset")
df_raw = ds["train"].to_pandas()  # convert to pandas DataFrame

# 3. Save the raw data to CSV
df_raw.to_csv(RAW_FILE, index=False, encoding="utf-8")
print(f"Raw HF dataset saved to {RAW_FILE}")
print("Columns detected:", df_raw.columns.tolist())

# 4. Keep only the columns we need: Email Text and Email Type
keep = ["Email Text", "Email Type"]
existing = [c for c in keep if c in df_raw.columns]
df = df_raw[existing].copy()

# 5. Rename columns to match our pipeline
df = df.rename(columns={
    "Email Text": "email_text",
    "Email Type": "label"
})

# 6. Convert string labels to integers (0 = safe, 1 = phishing)
if df["label"].dtype == object:
    df["label"] = df["label"].map({
        "Safe Email": 0,
        "Phishing Email": 1
    })

# 7. Save the cleaned data to CSV
df.to_csv(CLEAN_FILE, index=False, encoding="utf-8")
print(f"Clean HF dataset saved to {CLEAN_FILE}")
print(f"Total rows: {len(df)}")
print("Label distribution:")
print(df["label"].value_counts())
