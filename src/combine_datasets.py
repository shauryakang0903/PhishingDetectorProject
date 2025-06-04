import pandas as pd  # to work with CSV files and DataFrames
import os            # for file path operations

# 1. Define paths for the Zenodo and Hugging Face cleaned datasets
zenodo_path = r"C:\Users\shaur\PhishingDetectorProject\dataset\Phishing_validation_emails.csv"
hf_clean_path = r"C:\Users\shaur\PhishingDetectorProject\dataset\hf_phishing_clean.csv"

# 2. Define where to save the combined dataset and ensure the folder exists
out_path = r"C:\Users\shaur\PhishingDetectorProject\dataset\combined_phishing.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# 3. Load and rename the Zenodo dataset
df_zen = pd.read_csv(zenodo_path)
df_zen = df_zen.rename(columns={
    'Email Text': 'email_text',
    'Email Type': 'label'
})
# Map Zenodo labels to 0 (safe) and 1 (phishing)
df_zen['label'] = df_zen['label'].map({'Safe Email': 0, 'Phishing Email': 1})

# 4. Load the cleaned Hugging Face dataset (already has email_text and label)
df_hf = pd.read_csv(hf_clean_path)

# 5. Concatenate Zenodo and Hugging Face data, then shuffle
df_combined = pd.concat([df_zen, df_hf], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Save the combined dataset to CSV
df_combined.to_csv(out_path, index=False, encoding='utf-8')
print(f"Combined dataset saved to:\n   {out_path}")
print(f"   Total emails: {len(df_combined)}")
print("   Label distribution:")
print(df_combined['label'].value_counts())
