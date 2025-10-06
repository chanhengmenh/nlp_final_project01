from transformers import pipeline

class TransformerSentimentModel:
    """
    A wrapper for the Hugging Face sentiment analysis pipeline
    to ensure the model is loaded only once.
    """
    _instance = None

    def __new__(cls, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        if cls._instance is None:
            print("Loading transformer model...")
            cls._instance = super(TransformerSentimentModel, cls).__new__(cls)
            cls._instance.classifier = pipeline("sentiment-analysis", model=model_name)
            print("Model loaded.")
        return cls._instance

    def predict(self, texts):
        """Runs sentiment analysis on the input text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        return self.classifier(texts)

# Create a single instance to be used by other modules
transformer_model = TransformerSentimentModel()
 
if __name__ == "__main__":
    sample = "@VirginAmerica plus you've added commercials to the experience... tacky."
    results = transformer_model.predict(sample)
    print(results)
