import pandas as pd
import os
import glob

# Set the path to the Kaggle dataset directory
KAGGLE_DIR = r"C:\Users\shaur\PhishingDetectorProject\dataset\KAGGLE"

# Get a list of all CSV files in the Kaggle directory
csv_files = glob.glob(os.path.join(KAGGLE_DIR, "*.csv"))

# Raise an error if no CSV file is found
if not csv_files:
    raise FileNotFoundError(f"No CSV found in {KAGGLE_DIR}")

# Pick the first CSV file (assuming there’s only one file in the folder)
in_file = csv_files[0]
print("▶ Loading:", in_file)

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(in_file)
print("Columns found:", df.columns.tolist())

# Rename the columns if expected ones are present
if 'Email Text' in df.columns and 'Email Type' in df.columns:
    df = df.rename(columns={
        'Email Text': 'email_text',
        'Email Type': 'label'
    })

    # Convert label values to numeric format (0 for safe, 1 for phishing)
    df['label'] = df['label'].map({
        'Safe Email': 0,
        'Phishing Email': 1
    })
else:
    # Raise an error if expected columns are not present
    raise ValueError("Expected columns 'Email Text' and 'Email Type' in the Kaggle file.")

# Check if any labels failed to convert (i.e., became NaN)
if df['label'].isnull().any():
    bad = df.loc[df['label'].isnull(), 'label']
    raise ValueError(f"Unmapped labels found: {bad.unique()}")

# Save the cleaned dataset to a new CSV file
out_file = r"C:\Users\shaur\PhishingDetectorProject\dataset\kaggle_phishing_clean.csv"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
df.to_csv(out_file, index=False, encoding='utf-8')

# Print success message and show label distribution
print(f"✅ Cleaned Kaggle dataset saved to:\n   {out_file}")
print("Label distribution:\n", df['label'].value_counts())
