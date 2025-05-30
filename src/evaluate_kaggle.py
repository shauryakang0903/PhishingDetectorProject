

import sys, os, pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from preprocess import clean_text, URL_REGEX, SUSPICIOUS_KEYWORDS

model_path   = r"C:\Users\shaur\PhishingDetectorProject\models\weighted_nb_ensemble.pkl"
kaggle_csv   = r"C:\Users\shaur\PhishingDetectorProject\dataset\kaggle_phishing_clean.csv"


with open(model_path, "rb") as f:
    obj = pickle.load(f)
tfidf, mnb, gnb, alpha = obj["tfidf"], obj["mnb"], obj["gnb"], obj["alpha"]

print(f"Loaded ensemble with α = {alpha}\n")


df = pd.read_csv(kaggle_csv)
print(f" Loaded {len(df)} samples from {kaggle_csv}")


df["clean_text"] = df["email_text"].apply(clean_text)
df["word_count"] = df["clean_text"].str.split().apply(len)
df["url_count"] = df["email_text"].apply(lambda t: len(URL_REGEX.findall(str(t))))
df["suspicious_kw_count"] = df["clean_text"].apply(
    lambda s: sum(s.count(k) for k in SUSPICIOUS_KEYWORDS)
)


X_text = tfidf.transform(df["clean_text"])
X_num  = df[["word_count","url_count","suspicious_kw_count"]].values


p_text = mnb.predict_proba(X_text)[:,1]
p_num  = gnb.predict_proba(X_num)[:,1]
p_ens  = alpha * p_text + (1 - alpha) * p_num
y_pred = (p_ens >= 0.5).astype(int)


print("\n=== Kaggle External Evaluation ===\n")
print(classification_report(df["label"], y_pred, target_names=["Safe","Phishing"]))
print("Confusion Matrix:\n", confusion_matrix(df["label"], y_pred))
