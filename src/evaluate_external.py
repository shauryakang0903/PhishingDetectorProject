# src/evaluate_external.py

import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ─── 1. Make src importable ─────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─── 2. Import preprocessing utilities ────────────────────────────────────────
from preprocess import clean_text, URL_REGEX, SUSPICIOUS_KEYWORDS

# ─── 3. Load the trained ensemble ─────────────────────────────────────────────
model_path = r"C:\Users\shaur\PhishingDetectorProject\models\weighted_nb_ensemble.pkl"
with open(model_path, "rb") as f:
    obj = pickle.load(f)
tfidf, mnb, gnb, alpha = obj["tfidf"], obj["mnb"], obj["gnb"], obj["alpha"]

print(f"Loaded ensemble with α = {alpha}")

# ─── 4. Load external emails CSV ──────────────────────────────────────────────
csv_path = r"C:\Users\shaur\PhishingDetectorProject\dataset\external_emails.csv"
ext = pd.read_csv(csv_path)

# ─── 5. Preprocess external emails ────────────────────────────────────────────
# Clean text
ext["clean_text"] = ext["email_text"].apply(clean_text)
# Numeric features
ext["word_count"]          = ext["clean_text"].str.split().apply(len)
ext["url_count"]           = ext["email_text"].apply(lambda t: len(URL_REGEX.findall(str(t))))
ext["suspicious_kw_count"] = ext["clean_text"].apply(
    lambda s: sum(s.count(k) for k in SUSPICIOUS_KEYWORDS)
)

# ─── 6. Vectorize & assemble features ────────────────────────────────────────
X_text = tfidf.transform(ext["clean_text"])
X_num  = ext[["word_count","url_count","suspicious_kw_count"]].values

# ─── 7. Make predictions ──────────────────────────────────────────────────────
p_text = mnb.predict_proba(X_text)[:, 1]
p_num  = gnb.predict_proba(X_num)[:, 1]
p_ens  = alpha * p_text + (1 - alpha) * p_num
y_pred = (p_ens >= 0.5).astype(int)

# ─── 8. Evaluate ───────────────────────────────────────────────────────────────
print("\nClassification Report on External Data:\n")
print(classification_report(ext["label"], y_pred, target_names=["Safe","Phishing"]))

# Optional: show confusion matrix
cm = confusion_matrix(ext["label"], y_pred)
print("Confusion Matrix:\n", cm)
