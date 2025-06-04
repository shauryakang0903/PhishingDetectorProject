import sys, os, pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB

# Add the parent directory to the path so we can import the preprocess module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the preprocessing pipeline and default CSV path
from preprocess import preprocess_pipeline, DEFAULT_CSV_PATH

# Preprocess the dataset and get the TF-IDF vectorizer along with train-test split
X_train, X_test, y_train, y_test, tfidf = preprocess_pipeline(DEFAULT_CSV_PATH)

# Combine training and testing data into one dataset (we're training on full data now)
X_full = np.vstack([X_train.toarray(), X_test.toarray()])
y_full = np.concatenate([y_train, y_test])

# Separate the text-based features and the numeric features
X_text    = X_full[:, :-3]
X_numeric = X_full[:,  -3:]

# Train the Multinomial Naive Bayes model on text features
print("Training final MultinomialNB on full text features...")
mnb = MultinomialNB()
mnb.fit(X_text, y_full)

# Train the Gaussian Naive Bayes model on numeric features
print("Training final GaussianNB on full numeric features...")
gnb = GaussianNB()
gnb.fit(X_numeric, y_full)

# Store the trained models and the TF-IDF vectorizer in a dictionary
# Also save the alpha weight used for combining predictions
ensemble = {
    "tfidf" : tfidf,
    "mnb"   : mnb,
    "gnb"   : gnb,
    "alpha" : 0.8
}

# Create a directory named 'models' if it doesn't already exist
os.makedirs("models", exist_ok=True)

# Save the ensemble dictionary to a .pkl file so it can be loaded later for prediction
with open("models/weighted_nb_ensemble.pkl", "wb") as f:
    pickle.dump(ensemble, f)

# Print confirmation that everything was saved
print("\n Final ensemble trained & saved to models/weighted_nb_ensemble.pkl")
print("   Text model weight α =", ensemble["alpha"])
