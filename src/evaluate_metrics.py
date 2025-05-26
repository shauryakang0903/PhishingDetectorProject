

import sys, os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from preprocess import clean_text, URL_REGEX, SUSPICIOUS_KEYWORDS


ensemble_path = r"C:\Users\shaur\PhishingDetectorProject\models\weighted_nb_ensemble.pkl"
external_csv  = r"C:\Users\shaur\PhishingDetectorProject\dataset\external_emails.csv"


with open(ensemble_path, "rb") as f:
    obj = pickle.load(f)
tfidf, mnb, gnb, alpha = obj["tfidf"], obj["mnb"], obj["gnb"], obj["alpha"]

print(f"Loaded ensemble with α = {alpha}\n")


df_ext = pd.read_csv(external_csv)


df_ext["clean_text"] = df_ext["email_text"].apply(clean_text)
df_ext["word_count"] = df_ext["clean_text"].str.split().apply(len)
df_ext["url_count"] = df_ext["email_text"].apply(lambda t: len(URL_REGEX.findall(str(t))))
df_ext["suspicious_kw_count"] = df_ext["clean_text"].apply(
    lambda s: sum(s.count(k) for k in SUSPICIOUS_KEYWORDS)
)


X_text = tfidf.transform(df_ext["clean_text"])
X_num  = df_ext[["word_count","url_count","suspicious_kw_count"]].values


p_text = mnb.predict_proba(X_text)[:,1]
p_num  = gnb.predict_proba(X_num)[:,1]
p_ens  = alpha * p_text + (1 - alpha) * p_num
y_true = df_ext["label"].values
y_pred = (p_ens >= 0.5).astype(int)


metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "ROC AUC": roc_auc_score(y_true, p_ens)
}

print("=== Performance Metrics ===")
for name, value in metrics.items():
    print(f"{name:10s}: {value:.4f}")
print()


cm = confusion_matrix(y_true, y_pred)
print("=== Confusion Matrix ===")
print(cm)
print()


plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.xticks([0,1], ["Safe","Phishing"])
plt.yticks([0,1], ["Safe","Phishing"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center')
plt.tight_layout()
plt.show()


fpr, tpr, _ = roc_curve(y_true, p_ens)
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.show()


precision, recall, _ = precision_recall_curve(y_true, p_ens)
plt.figure()
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()
