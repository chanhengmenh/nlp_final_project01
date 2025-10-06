import argparse
import joblib
from src.config import MODEL_PATH, VECTORIZER_PATH
from src.preprocess import preprocess_text
from src.transformer import transformer_model

# --- Efficiently load classical models once ---
try:
    classical_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    # Handle case where models are not yet trained
    classical_model, vectorizer = None, None

def predict_sentiment_classical(tweet: str) -> str:
    """
    Predict sentiment using the classical ML pipeline (TF-IDF + Logistic Regression).
    """
    # Preprocess + vectorize text
    processed = preprocess_text(tweet)
    tweet_vec = vectorizer.transform([processed])

    # Predict
    prediction = classical_model.predict(tweet_vec)
    return prediction[0]

def predict_sentiment_transformer(tweet: str) -> str:
    """
    Predict sentiment using transformer-based model (RoBERTa).
    """
    results = transformer_model.predict(tweet)
    label = results[0]["label"]
    score = results[0]["score"]
    return f"{label} (confidence: {score:.2f})"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the sentiment of a tweet.")
    parser.add_argument("tweet", type=str, help="The tweet to analyze.")
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Use transformer-based model (RoBERTa) instead of classical ML."
    )

    args = parser.parse_args()

    if not args.transformer and (classical_model is None or vectorizer is None):
        print("Error: Classical model files not found. Please train the model first by running 'python -m src.train'")
        exit(1)

    if args.transformer:
        sentiment = predict_sentiment_transformer(args.tweet)
        print(f"[Transformer] Predicted sentiment: {sentiment}")
    else:
        sentiment = predict_sentiment_classical(args.tweet)
        print(f"[Classical ML] Predicted sentiment: {sentiment}")
