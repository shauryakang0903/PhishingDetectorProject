

import pandas as pd
import os


zenodo_path = r"C:\Users\shaur\PhishingDetectorProject\dataset\Phishing_validation_emails.csv"
hf_clean_path = r"C:\Users\shaur\PhishingDetectorProject\dataset\hf_phishing_clean.csv"


out_path = r"C:\Users\shaur\PhishingDetectorProject\dataset\combined_phishing.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)


df_zen = pd.read_csv(zenodo_path)
df_zen = df_zen.rename(columns={
    'Email Text': 'email_text',
    'Email Type': 'label'
})
df_zen['label'] = df_zen['label'].map({'Safe Email': 0, 'Phishing Email': 1})


df_hf = pd.read_csv(hf_clean_path)


df_combined = pd.concat([df_zen, df_hf], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)


df_combined.to_csv(out_path, index=False, encoding='utf-8')
print(f" Combined dataset saved to:\n   {out_path}")
print(f"   Total emails: {len(df_combined)}")
print("    Label distribution:")
print(df_combined['label'].value_counts())
