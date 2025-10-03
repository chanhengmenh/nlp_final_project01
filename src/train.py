import pandas as pd
import joblib
import nltk
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import RAW_DATA_PATH, MODEL_PATH, VECTORIZER_PATH, RESULTS_DIR, SEED
from src.preprocess import preprocess_text
from src.evaluate import evaluate_model

# Ensure nltk downloads are present
def download_nltk_data():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

def train_model():
    """Train sentiment analysis model with TF-IDF + Logistic Regression."""
    download_nltk_data()

    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Preprocess text
    df["processed_text"] = df["text"].apply(preprocess_text)
    X = df["processed_text"]
    y = df["airline_sentiment"]

    # Split into train, validation, test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    # Define pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    results = evaluate_model(pipeline, X_test, y_test, save_path=RESULTS_DIR)

    # Save model and vectorizer separately
    joblib.dump(pipeline.named_steps["clf"], MODEL_PATH)
    joblib.dump(pipeline.named_steps["tfidf"], VECTORIZER_PATH)

    # Save metrics to JSON
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")
    print(f"Evaluation report saved to {RESULTS_DIR}/metrics.json")

if __name__ == "__main__":
    train_model()
