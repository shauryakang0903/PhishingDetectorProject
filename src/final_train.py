

import sys, os, pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess import preprocess_pipeline, DEFAULT_CSV_PATH


X_train, X_test, y_train, y_test, tfidf = preprocess_pipeline(DEFAULT_CSV_PATH)
X_full = np.vstack([X_train.toarray(), X_test.toarray()])
y_full = np.concatenate([y_train, y_test])


X_text    = X_full[:, :-3]
X_numeric = X_full[:,  -3:]


print("Training final MultinomialNB on full text features...")
mnb = MultinomialNB()
mnb.fit(X_text, y_full)

print("Training final GaussianNB on full numeric features...")
gnb = GaussianNB()
gnb.fit(X_numeric, y_full)


ensemble = {
    "tfidf" : tfidf,
    "mnb"   : mnb,
    "gnb"   : gnb,
    "alpha" : 0.8
}

os.makedirs("models", exist_ok=True)
with open("models/weighted_nb_ensemble.pkl", "wb") as f:
    pickle.dump(ensemble, f)

print("\n Final ensemble trained & saved to models/weighted_nb_ensemble.pkl")
print("   Text model weight α =", ensemble["alpha"])
