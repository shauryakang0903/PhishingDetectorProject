

import sys, os
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit
from preprocess import preprocess_pipeline, DEFAULT_CSV_PATH


X_tr, X_te, y_tr, y_te, tfidf = preprocess_pipeline(DEFAULT_CSV_PATH)
X_full = np.vstack([X_tr.toarray(), X_te.toarray()])
y_full = np.concatenate([y_tr, y_te])


X_text    = X_full[:, :-3]
X_numeric = X_full[:,  -3:]


mnb = MultinomialNB()
gnb = GaussianNB()


sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)


alphas = np.linspace(0, 1, 11)
best_alpha, best_score = None, 0.0

for alpha in alphas:
    scores = []
    for train_idx, test_idx in sss.split(X_full, y_full):
        
        mnb.fit(X_text[train_idx], y_full[train_idx])
        gnb.fit(X_numeric[train_idx], y_full[train_idx])

        
        p_text = mnb.predict_proba(X_text[test_idx])[:, 1]
        p_num  = gnb.predict_proba(X_numeric[test_idx])[:, 1]
        p_ens  = alpha * p_text + (1 - alpha) * p_num

        
        preds = (p_ens >= 0.5).astype(int)
        scores.append((preds == y_full[test_idx]).mean())

    mean_score = np.mean(scores)
    print(f"α = {alpha:.1f} → accuracy = {mean_score:.4f}")
    if mean_score > best_score:
        best_score, best_alpha = mean_score, alpha

print("\n🔎 Best α:", best_alpha, "with CV accuracy:", best_score)
