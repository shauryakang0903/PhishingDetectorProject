# src/clean_kaggle.py

import pandas as pd
import os
import glob

# 1. Locate the CSV in your KAGGLE folder
KAGGLE_DIR = r"C:\Users\shaur\PhishingDetectorProject\dataset\KAGGLE"
csv_files = glob.glob(os.path.join(KAGGLE_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV found in {KAGGLE_DIR}")
in_file = csv_files[0]
print("▶ Loading:", in_file)

# 2. Load it
df = pd.read_csv(in_file)
print("Columns found:", df.columns.tolist())

# 3. Handle the Zenodo-style columns
if 'Email Text' in df.columns and 'Email Type' in df.columns:
    df = df.rename(columns={
        'Email Text': 'email_text',
        'Email Type': 'label'
    })
    df['label'] = df['label'].map({
        'Safe Email': 0,
        'Phishing Email': 1
    })
else:
    raise ValueError("Expected columns 'Email Text' and 'Email Type' in the Kaggle file.")

# 4. Sanity-check
if df['label'].isnull().any():
    bad = df.loc[df['label'].isnull(), 'label']
    raise ValueError(f"Unmapped labels found: {bad.unique()}")

# 5. Save cleaned file
out_file = r"C:\Users\shaur\PhishingDetectorProject\dataset\kaggle_phishing_clean.csv"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
df.to_csv(out_file, index=False, encoding='utf-8')
print(f"✅ Cleaned Kaggle dataset saved to:\n   {out_file}")
print("Label distribution:\n", df['label'].value_counts())
