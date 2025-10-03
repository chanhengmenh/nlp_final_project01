# Tweet Sentiment Analysis

This project provides a comprehensive pipeline for sentiment analysis on tweets, classifying them as positive, negative, or neutral. It features two distinct models: a classical machine learning approach (TF-IDF with Logistic Regression) and a modern transformer-based model (RoBERTa).

## Features

- **Dual Model Architecture**: Choose between a fast, classical model and a powerful, deep learning-based transformer model.
- **End-to-End Pipeline**: Scripts for preprocessing, training, evaluation, and prediction are included.
- **Reproducibility**: Ensures consistent results with a fixed random seed.
- **In-depth Evaluation**: Generates a classification report, confusion matrix, and performance metrics.
- **Easy-to-Use**: Simple command-line interface for training and prediction.

## Project Structure

```
.final_project/
├───.gitignore
├───README.md
├───requirements.txt
├───dataset/
│   └───raw/
│       ├───Restaurant_Reviews.tsv
│       └───Tweets.csv
├───models/
│   ├───sentiment_model.pkl
│   └───vectorizer.pkl
├───notebooks/
│   └───sentiment_analysis_exploration.ipynb
├───results/
│   ├───confusion_matrix.png
│   └───classification_report.csv
└───src/
    ├───__init__.py
    ├───config.py           # Configuration file for paths and parameters
    ├───evaluate.py         # Model evaluation script
    ├───predict.py          # Script for making predictions
    ├───preprocess.py       # Text preprocessing functions
    ├───train.py            # Model training script
    └───transformer.py      # Transformer model inference functions
```

## Dataset

The project uses the "Tweets.csv" dataset, which contains tweets about various airlines and their corresponding sentiment labels (positive, negative, neutral).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chanhengmenh/nlp_final_project01.git
    cd nlp_final_project01
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv nlp_final_project01
    source nlp_final_project01/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train the Classical Model

To train the sentiment analysis model (TF-IDF + Logistic Regression), run the following command from the root directory:

```bash
python -m src.train
```

This script will:
- Load and preprocess the `Tweets.csv` dataset.
- Train a Logistic Regression model.
- Save the trained model and vectorizer to the `models/` directory.
- Save evaluation metrics to the `results/` directory.

### 2. Predict Sentiment

To predict the sentiment of a new tweet, use the `predict.py` script. You can choose between the classical model and the transformer model.

**Using the Classical Model:**

```bash
python -m src.predict "Your tweet text here"
```

**Example:**
```bash
python -m src.predict "@VirginAmerica plus you've added commercials to the experience... tacky."
# Output: [Classical ML] Predicted sentiment: negative
```

**Using the Transformer Model:**

Use the `--transformer` flag to leverage the RoBERTa model.

```bash
python -m src.predict --transformer "Your tweet text here"
```

**Example:**
```bash
python -m src.predict --transformer "@VirginAmerica plus you've added commercials to the experience... tacky."
# Output: [Transformer] Predicted sentiment: negative (confidence: 0.84)
```

### 3. Exploratory Data Analysis

To explore the dataset, you can use the Jupyter Notebook located at `notebooks/sentiment_analysis_exploration.ipynb`.

## Model Details

### Classical Model

- **Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF) with n-grams (1, 2).
- **Classifier**: Logistic Regression.

### Transformer Model

- **Model**: `cardiffnlp/twitter-roberta-base-sentiment`, a RoBERTa model fine-tuned for sentiment analysis on Twitter data.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License.
