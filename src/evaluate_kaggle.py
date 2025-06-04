import sys, os, pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Add the parent directory to the system path to import from preprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import necessary preprocessing functions and constants
from preprocess import clean_text, URL_REGEX, SUSPICIOUS_KEYWORDS

# Paths to the trained model and Kaggle dataset
model_path   = r"C:\Users\shaur\PhishingDetectorProject\models\weighted_nb_ensemble.pkl"
kaggle_csv   = r"C:\Users\shaur\PhishingDetectorProject\dataset\kaggle_phishing_clean.csv"

# Load the trained ensemble model (includes tf-idf, MultinomialNB, GaussianNB, and alpha weight)
with open(model_path, "rb") as f:
    obj = pickle.load(f)
tfidf, mnb, gnb, alpha = obj["tfidf"], obj["mnb"], obj["gnb"], obj["alpha"]

print(f"Loaded ensemble with α = {alpha}\n")

# Load Kaggle dataset (already cleaned and preprocessed)
df = pd.read_csv(kaggle_csv)
print(f" Loaded {len(df)} samples from {kaggle_csv}")

# Preprocess the emails: clean text, count words, URLs, and suspicious keywords
df["clean_text"] = df["email_text"].apply(clean_text)
df["word_count"] = df["clean_text"].str.split().apply(len)
df["url_count"] = df["email_text"].apply(lambda t: len(URL_REGEX.findall(str(t))))
df["suspicious_kw_count"] = df["clean_text"].apply(
    lambda s: sum(s.count(k) for k in SUSPICIOUS_KEYWORDS)
)

# Vectorize email text using the saved TF-IDF vectorizer
X_text = tfidf.transform(df["clean_text"])
# Create numeric feature array (word count, URL count, keyword count)
X_num  = df[["word_count", "url_count", "suspicious_kw_count"]].values

# Get probabilities from both models
p_text = mnb.predict_proba(X_text)[:,1]  # MultinomialNB prediction on text
p_num  = gnb.predict_proba(X_num)[:,1]   # GaussianNB prediction on numeric features

# Combine both model predictions using weighted average (alpha)
p_ens  = alpha * p_text + (1 - alpha) * p_num
# Final prediction: classify as 1 (phishing) if probability ≥ 0.5
y_pred = (p_ens >= 0.5).astype(int)

# Show classification results and confusion matrix
print("\n=== Kaggle External Evaluation ===\n")
print(classification_report(df["label"], y_pred, target_names=["Safe", "Phishing"]))
print("Confusion Matrix:\n", confusion_matrix(df["label"], y_pred))
