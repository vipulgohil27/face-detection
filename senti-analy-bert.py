import torch
from transformers import pipeline

# Load DistilBERT sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def detect_extreme_negative(text):
    """Detect if a given text is extremely negative."""
    result = sentiment_analyzer(text)[0]  # Get the first result
    label = result['label']
    score = result['score']

    # Consider "Negative" sentiment with high confidence as extreme negativity
    if label == "NEGATIVE" and score > 0.95:  # Adjust threshold as needed
        return f"🔥 EXTREMELY NEGATIVE: {score:.4f}"
    elif label == "NEGATIVE":
        return f"⚠️ Negative but not extreme: {score:.4f}"
    else:
        return f"✅ Positive sentiment: {score:.4f}"

# Example texts
texts = [
    "I hate everything about this! It's the worst experience ever!",
    "This is really bad. I'm very disappointed.",
    "I'm not happy, but it's not terrible.",
    "I absolutely love it! Best thing ever!",
    "my internet not working",
    "blinking red light",
    "this sucks",
    "bad internet",
    "fix my problem",
    "numan agent",
    "talk to agent",
    "talk to person"
]

# Analyze sentiment
for text in texts:
    print(f"Text: {text}\nResult: {detect_extreme_negative(text)}\n")
