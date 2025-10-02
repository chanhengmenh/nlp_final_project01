# NLP Final Project

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project aims to classify the sentiment of tweets as positive, negative, or neutral. It provides a complete workflow for data preprocessing, feature extraction, model training, and prediction.

## Features

-   **Text Preprocessing Pipeline**: Cleans and prepares tweet text for modeling.
-   **Sentiment Classification**: Classifies tweets into positive, negative, or neutral categories.
-   **Trainable Model**: A script to train the sentiment analysis model from scratch.
-   **Prediction Script**: A command-line interface to predict the sentiment of new tweets.

## Project Structure

```
.
├── dataset
│   └── raw
│       ├── Restaurant_Reviews.tsv
│       └── Tweets.csv
├── models
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
├── notebooks
│   └── sentiment_analysis_exploration.ipynb
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── .gitignore
└── requirements.txt
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

Run the following commands in a Python interpreter to download the necessary NLTK datasets:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## Usage

### Training the Model

To train the sentiment analysis model, run the `train.py` script. This will preprocess the raw data, train the model, and save the artifacts in the `models/` directory.

```bash
python src/train.py
```

### Making a Prediction

To make a sentiment prediction on a new tweet, use the `predict.py` script with the `--text` argument.

```bash
python src/predict.py --text "This is a great movie!"
```

## Data

The primary dataset used for this project is `Tweets.csv`, which contains tweets about major U.S. airlines. An additional dataset, `Restaurant_Reviews.tsv`, is also available in the `dataset/raw` directory.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or find any bugs.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
