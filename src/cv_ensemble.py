

import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocess import preprocess_pipeline, DEFAULT_CSV_PATH

def run_stratified_cv(n_splits=5, alpha=0.8, random_state=42):
    
    X_train, X_test, y_train, y_test, tfidf = preprocess_pipeline(DEFAULT_CSV_PATH)
    X_full = np.vstack([X_train.toarray(), X_test.toarray()])
    y_full = np.concatenate([y_train, y_test])

    
    X_text    = X_full[:, :-3]
    X_numeric = X_full[:,  -3:]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    
    accs = []; precs = []; recs = []; f1s = []; aucs = []

    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y_full), 1):
       
        mnb = MultinomialNB().fit(X_text[train_idx], y_full[train_idx])
        gnb = GaussianNB().fit(X_numeric[train_idx], y_full[train_idx])

        
        p_text = mnb.predict_proba(X_text[test_idx])[:, 1]
        p_num  = gnb.predict_proba(X_numeric[test_idx])[:, 1]
        p_ens  = alpha * p_text + (1 - alpha) * p_num
        y_pred = (p_ens >= 0.5).astype(int)

       
        accs.append(accuracy_score(y_full[test_idx], y_pred))
        precs.append(precision_score(y_full[test_idx], y_pred))
        recs.append(recall_score(y_full[test_idx], y_pred))
        f1s.append(f1_score(y_full[test_idx], y_pred))
        aucs.append(roc_auc_score(y_full[test_idx], p_ens))

        print(f"Fold {fold}: "
              f"Acc={accs[-1]:.4f}, "
              f"P={precs[-1]:.4f}, "
              f"R={recs[-1]:.4f}, "
              f"F1={f1s[-1]:.4f}, "
              f"AUC={aucs[-1]:.4f}")

    
    print("\n=== CV Summary ===")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"F1 Score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"ROC AUC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

if __name__ == "__main__":
    run_stratified_cv()
