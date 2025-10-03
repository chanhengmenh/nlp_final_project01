import os
import random
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the raw data
DATA_DIR = os.path.join(ROOT_DIR, "dataset")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "Tweets.csv")

# Path to the directory where the model will be saved
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Results directory (for metrics, confusion matrix, logs)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
