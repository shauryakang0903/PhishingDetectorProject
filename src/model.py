import sys
import os

# 📍 Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess import preprocess_pipeline, DEFAULT_CSV_PATH
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

# 🧹 Load and preprocess data
X_train, X_test, y_train, y_test, tfidf = preprocess_pipeline(DEFAULT_CSV_PATH)

# 🔀 Merge train and test back for full CV
X = np.vstack([X_train.toarray(), X_test.toarray()])
y = np.concatenate([y_train, y_test])

# ➗ Split back into text and numeric features
X_text = X[:, :-3]       # TF-IDF features
X_numeric = X[:, -3:]    # Custom numeric features

# ✅ Text model pipeline: TF-IDF already applied, just use MultinomialNB
text_model = make_pipeline(MultinomialNB())

# ✅ Numeric model pipeline: StandardScaler + GaussianNB
numeric_model = make_pipeline(StandardScaler(), GaussianNB())

# 📊 5-Fold Cross-validation
print("🔍 Evaluating with 5-Fold Cross-Validation...\n")

# Text-based model evaluation
scores_text = cross_val_score(text_model, X_text, y, cv=5, scoring='accuracy')
print("📘 Text Model (MultinomialNB + TF-IDF)")
print(f"  → Accuracy: {scores_text.mean():.4f} ± {scores_text.std():.4f}\n")

# Numeric-based model evaluation
scores_num = cross_val_score(numeric_model, X_numeric, y, cv=5, scoring='accuracy')
print("📗 Numeric Model (GaussianNB + Scaled Features)")
print(f"  → Accuracy: {scores_num.mean():.4f} ± {scores_num.std():.4f}")
