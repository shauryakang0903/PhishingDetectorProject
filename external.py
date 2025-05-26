# generate_external.py

import csv
import os

# ─── 1. Define your external email samples here ──────────────────────────────
# Each tuple is: (raw_email_text, label) where label is 1 for phishing, 0 for safe.
external_emails = [
    ("Dear user, your account has been suspended. Click http://fake-bank.com to verify.", 1),
    ("Hello John, please find attached the report from last week. Regards, Jane.", 0),
    ("Urgent: Your password will expire soon. Login at www.security-update.com to renew.", 1),
    ("Team, the meeting is moved to 3 PM tomorrow. Thank you.", 0),
    # Add your own examples below:
    # ("[Your example phishing email here]", 1),
    # ("[Your example safe email here]", 0),
]

# ─── 2. Specify your project’s dataset directory ──────────────────────────────
DATASET_DIR = r"C:\Users\shaur\PhishingDetectorProject\dataset"

# Ensure the dataset folder exists
os.makedirs(DATASET_DIR, exist_ok=True)

# ─── 3. Write to CSV ───────────────────────────────────────────────────────────
csv_path = os.path.join(DATASET_DIR, "external_emails.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(["email_text", "label"])
    # Rows
    for text, label in external_emails:
        writer.writerow([text, label])

print(f"✅ external_emails.csv generated with {len(external_emails)} samples at:\n   {csv_path}")
