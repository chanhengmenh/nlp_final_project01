from transformers import pipeline

def transformer_sentiment(texts, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
    """
    Runs transformer-based sentiment analysis on input texts.
    
    Args:
        texts (str or list): A string or list of texts.
        model_name (str): Pretrained model name.
    
    Returns:
        List of predictions with labels and scores.
    """
    classifier = pipeline("sentiment-analysis", model=model_name)

    if isinstance(texts, str):
        texts = [texts]

    return classifier(texts)

if __name__ == "__main__":
    sample = "@VirginAmerica plus you've added commercials to the experience... tacky."
    print(transformer_sentiment(sample))
