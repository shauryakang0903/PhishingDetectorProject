# src/clean_hf.py

import pandas as pd
import os

# 1. Paths (adjust if yours differ)
HF_RAW   = r"C:\Users\shaur\PhishingDetectorProject\dataset\hf_phishing.csv"
HF_CLEAN = r"C:\Users\shaur\PhishingDetectorProject\dataset\hf_phishing_clean.csv"

# 2. Load raw HF dataset
df = pd.read_csv(HF_RAW)

# 3. Keep only the columns we need (ignore any others)
needed = ['Email Text', 'Email Type']
# If the CSV has slightly different headers, adjust those names here
df = df[[col for col in needed if col in df.columns]]

# 4. Rename them
df = df.rename(columns={
    'Email Text': 'email_text',
    'Email Type': 'label'
})

# 5. Convert labels to 0/1 if they’re strings
if df['label'].dtype == object:
    df['label'] = df['label'].map({
        'Safe Email': 0,
        'Phishing Email': 1
    })
    # If they use other strings like 'legit' / 'phish', add them here

# 6. Save the cleaned file
os.makedirs(os.path.dirname(HF_CLEAN), exist_ok=True)
df.to_csv(HF_CLEAN, index=False, encoding='utf-8')
print(f"✅ Cleaned HF dataset saved to:\n   {HF_CLEAN}\nTotal rows: {len(df)}")
