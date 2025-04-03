import nltk
import logging
import random
import string
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from typing import List, Dict , Any
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary datasets (only once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('vader_lexicon',quiet=True)

# Load stopwords once to optimize performance
STOP_WORDS = set(stopwords.words("english"))

def preprocess_text(text: str) -> List[str]:
    """
    Tokenizes text, removes stopwords and punctuation, and normalizes words.
    """
    try:
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in STOP_WORDS]
        lemmatizer = WordNetLemmatizer()
        lemmatizer_tokens=lemmatizer.lemmatize(tokens)
        print(lemmatizer_tokens)
        processed_text= ' '.join(lemmatizer_tokens)
        return processed_text
    except Exception as e:
        logging.error(f"Error in text preprocessing: {e}")
        return []

def extract_features(words: List[str]) -> Dict[str, bool]:
    """
    Extracts features by treating each word as a feature.
    """
    return {word: True for word in words}

def load_dataset_from_csv(csv_path: str) -> List[tuple]:
    """
    Loads dataset from a CSV file with 'text' and 'label' columns.
    """
    dataset = []
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            words = preprocess_text(row['text'])
            if words:
                features = extract_features(words)
                dataset.append((features, row['label']))
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
    return dataset

# Load training data from CSV
data = load_dataset_from_csv('C:\\Users\\vipul\\PycharmProjects\\PythonProject\\data\\input-txt.csv')

# Shuffle and split dataset
random.shuffle(data)
train_size = int(0.8 * len(data))
train_data, test_data = data[:train_size], data[train_size:]

# Train the classifier
#classifier = NaiveBayesClassifier.train(train_data)

# Evaluate the classifier
#accuracy = nltk.classify.accuracy(classifier, test_data)
#logging.info(f"Model Accuracy: {accuracy:.2f}")
#classifier.show_most_informative_features(5)

def predict_sentiment(sentence: str) -> str:
    """
    Predicts the sentiment of a given sentence.
    """
    words = preprocess_text(sentence)
    print(words)
    #features = extract_features(words)
    sia=SentimentIntensityAnalyzer()
    senti_score=sia.polarity_scores(words)
    print(senti_score)
    return "unknown"#classifier.classify(words) if words else "unknown"

def test_model_from_csv(csv_path: str):
    """
    Loads test sentences from a CSV file and predicts sentiment.
    """
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            sentiment = predict_sentiment(row['text'])
            logging.info(f"Sentence: {row['text']}\nPredicted Sentiment: {sentiment}\n")
    except Exception as e:
        logging.error(f"Error loading test dataset: {e}")

# Test predictions from CSV
#test_model_from_csv('test_sentences.csv')
if __name__ == "__main__":
    test_model_from_csv("C:\\Users\\vipul\\PycharmProjects\\PythonProject\\data\\input-txt.csv")
