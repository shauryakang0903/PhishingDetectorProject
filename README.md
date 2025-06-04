                            Phishing Email Detection Using Hybrid Naive Bayes Ensemble
This project implements a phishing email detector using an ensemble of Multinomial Naive Bayes (MNB) and Gaussian Naive Bayes (GNB). Text features are vectorized with TF-IDF and handled by MNB, while numeric features (word count, URL count, suspicious keyword count) are handled by GNB. The final prediction is a weighted combination of both models.



Installation

Ensure Python 3.8 or higher is installed. Create and activate a virtual environment (optional but recommended). Install required packages by running:
pip install -r requirements.txt

Script Execution Order

download_hf_dataset.py

Downloads the Hugging Face phishing dataset and saves raw data (hf_phishing_raw.csv) and cleaned data (hf_phishing_clean.csv) in dataset/.

combine_datasets.py

Loads Zenodo’s Phishing_validation_emails.csv and the cleaned Hugging Face dataset. Renames columns and maps labels to numeric. Combines both datasets into combined_phishing.csv and shuffles it.

preprocess.py (used internally)

Defines functions for loading data, cleaning text, extracting numeric features, vectorizing text with TF-IDF, and splitting into train/test sets. This file is imported by other scripts and is not executed directly.

ensemble_tuning.py

Uses the combined dataset and runs StratifiedShuffleSplit cross-validation. Tests a grid of alpha values (from 0.0 to 1.0) to determine the optimal weight for combining MNB and GNB outputs. Prints the accuracy for each alpha and reports the best alpha.

cv_ensemble.py

Loads the combined dataset via preprocess.py. Performs 5-fold StratifiedKFold cross-validation using the chosen alpha (default 0.8). Trains MNB on text features and GNB on numeric features for each fold. Computes and prints Accuracy, Precision, Recall, F1-score, and ROC AUC for each fold, as well as the averaged metrics.

final_train.py

Preprocesses the combined dataset in its entirety. Trains a final MultinomialNB model on all TF-IDF text features and a final GaussianNB model on all numeric features. Packages the TF-IDF vectorizer, both trained models, and the chosen alpha into a dictionary. Saves the ensemble object as models/weighted_nb_ensemble.pkl.

clean_kaggle.py

Searches the dataset/KAGGLE/ folder for a Kaggle CSV file. Loads and renames “Email Text” to email_text and “Email Type” to label. Maps “Safe Email” to 0 and “Phishing Email” to 1. Saves the cleaned Kaggle dataset as dataset/kaggle_phishing_clean.csv.

evaluate_kaggle.py

Loads the trained ensemble model (models/weighted_nb_ensemble.pkl). Reads the cleaned Kaggle test set (kaggle_phishing_clean.csv). Preprocesses each email (clean text, compute numeric features). Vectorizes text with the saved TF-IDF object and extracts numeric features. Obtains phishing probabilities from MNB and GNB, combines them using alpha, and thresholds at 0.5. Prints a classification report (Precision, Recall, F1-score for each class) and the confusion matrix.

Expected Outputs

download_hf_dataset.py

dataset/hf_phishing_raw.csv

dataset/hf_phishing_clean.csv

combine_datasets.py

dataset/combined_phishing.csv (shuffled, combined dataset)

Printed total number of combined emails and label distribution.

ensemble_tuning.py

Printed accuracy for each alpha value tested. Reports the best alpha with its corresponding accuracy.

cv_ensemble.py

Printed per-fold metrics (Accuracy, Precision, Recall, F1, ROC AUC). Printed overall averaged metrics and standard deviations across folds.

final_train.py

models/weighted_nb_ensemble.pkl (serialized ensemble containing TF-IDF, MNB, GNB, and alpha). Printed confirmation of saved model and alpha value used.

clean_kaggle.py

dataset/kaggle_phishing_clean.csv (cleaned Kaggle dataset). Printed total sample count and label distribution.

evaluate_kaggle.py

Printed classification report and confusion matrix on the Kaggle test set.

Requirements

Create a file named requirements.txt containing:

pandas==2.2.2

numpy==1.26.4

scipy==1.12.3

scikit-learn==1.4.2

datasets==2.18.0

Then run:

pip install -r requirements.txt

Notes
The preprocess.py script defines the full preprocessing pipeline and is used by multiple other scripts. Do not run it directly.

Ensure that the dataset/ directory contains the original Zenodo CSV (Phishing_validation_emails.csv) before running any scripts.

When evaluating on the Kaggle dataset, confirm that dataset/KAGGLE/ contains exactly one CSV file. The clean_kaggle.py script will process the first CSV it finds.

The chosen alpha (0.8) reflects the optimal balance between text and numeric model outputs as determined by cross-validation.
