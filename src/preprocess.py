

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import scipy.sparse as sp


DEFAULT_CSV_PATH = r"C:\Users\shaur\PhishingDetectorProject\dataset\combined_phishing.csv"



def load_and_rename(csv_path: str) -> pd.DataFrame:
    """
    Loads the CSV and standardizes column names and label values.
    Supports:
      - Original Zenodo format: 'Email Text' (str), 'Email Type' ('Safe Email'/'Phishing Email')
      - Combined format: 'email_text', 'label' (0/1)
    """
    df = pd.read_csv(csv_path)

    # Case A: original Zenodo columns
    if 'Email Text' in df.columns and 'Email Type' in df.columns:
        df = df.rename(columns={
            'Email Text': 'email_text',
            'Email Type': 'label'
        })
        df['label'] = df['label'].map({
            'Safe Email': 0,
            'Phishing Email': 1
        })

    # Case B: already cleaned combined dataset
    elif 'email_text' in df.columns and 'label' in df.columns:
        # If labels are strings, map them; if ints, leave as is
        if df['label'].dtype == object:
            df['label'] = df['label'].map({
                'Safe Email': 0,
                'Phishing Email': 1,
                '0': 0,
                '1': 1
            })
    else:
        raise ValueError(
            "Unexpected columns: found "
            f"{df.columns.tolist()}"
        )

    
    if df['label'].isnull().any():
        raise ValueError(
            "Unexpected label values found after mapping: "
            f"{df['label'].unique()}"
        )

    return df




URL_REGEX        = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_REGEX   = re.compile(r'<[^>]+>')
EMAIL_ADDR_REGEX = re.compile(r'\S+@\S+')

def clean_text(text: str) -> str:
    text = str(text)
    text = HTML_TAG_REGEX.sub(' ', text)
    text = URL_REGEX.sub(' ', text)
    text = EMAIL_ADDR_REGEX.sub(' ', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



SUSPICIOUS_KEYWORDS = [
    'verify','account','login','update','urgent',
    'password','bank','social','security','click'
]

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df['clean_text'] = df['email_text'].apply(clean_text)
    df['word_count']  = df['clean_text'].apply(lambda s: len(s.split()))
    df['url_count']   = df['email_text'].apply(lambda t: len(URL_REGEX.findall(str(t))))
    df['suspicious_kw_count'] = df['clean_text'].apply(
        lambda s: sum(s.count(k) for k in SUSPICIOUS_KEYWORDS)
    )
    return df[['clean_text','word_count','url_count','suspicious_kw_count','label']]



def vectorize_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    X_text = df['clean_text']
    y      = df['label']

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        stop_words='english'
    )
    X_tfidf = tfidf.fit_transform(X_text)

    X_num = df[['word_count','url_count','suspicious_kw_count']].values
    X     = sp.hstack([X_tfidf, X_num])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, tfidf



def preprocess_pipeline(
    csv_path: str = DEFAULT_CSV_PATH,
    test_size: float = 0.2,
    random_state: int = 42
):
    df = load_and_rename(csv_path)
    df = extract_features(df)
    return vectorize_and_split(df, test_size=test_size, random_state=random_state)



if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tfidf = preprocess_pipeline()
    print("Preprocessing complete")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test  shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test  shape: {y_test.shape}")
