import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer data
nltk.download('punkt')

# Define possible troubleshooting intents
troubleshooting_intents = [
    "slow_internet",
    "no_connection",
    "router_issue",
    "website_blocked",
    "wifi_dropping",
    "high_ping",
    "modem_not_working",
    "other"
]

# Load DistilBERT tokenizer and model (Need fine-tuned model for better accuracy)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                            num_labels=len(troubleshooting_intents))


# Function to preprocess input text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join(tokens)


# Function to predict troubleshooting intent
def predict_issue(text):
    preprocessed_text = preprocess(text)
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return troubleshooting_intents[predicted_class]


# Example usage
user_input = "light blinking red"
predicted_issue = predict_issue(user_input)
print(f"Predicted Issue: {predicted_issue}")
