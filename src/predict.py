import argparse
import joblib
from src.config import MODEL_PATH, VECTORIZER_PATH
from src.preprocess import preprocess_text
from src.transformer import transformer_sentiment

def predict_sentiment_classical(tweet: str) -> str:
    """
    Predict sentiment using the classical ML pipeline (TF-IDF + Logistic Regression).
    """
    # Load model + vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Preprocess + vectorize
    processed = preprocess_text(tweet)
    tweet_vec = vectorizer.transform([processed])

    # Predict
    prediction = model.predict(tweet_vec)
    return prediction[0]

def predict_sentiment_transformer(tweet: str) -> str:
    """
    Predict sentiment using transformer-based model (RoBERTa).
    """
    results = transformer_sentiment(tweet)
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

    if args.transformer:
        sentiment = predict_sentiment_transformer(args.tweet)
        print(f"[Transformer] Predicted sentiment: {sentiment}")
    else:
        sentiment = predict_sentiment_classical(args.tweet)
        print(f"[Classical ML] Predicted sentiment: {sentiment}")
