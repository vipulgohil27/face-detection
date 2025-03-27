import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer data
nltk.download('punkt')

# Define intents
intents = ["greeting", "order_status", "complaint", "goodbye","trobleshooting"]

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(intents))

# Function to preprocess input text
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join(tokens)

# Function to predict intent
def predict_intent(text):
    preprocessed_text = preprocess(text)
    inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    print(f"predicted_class{predicted_class}")
    return intents[predicted_class]

# Example usage
user_input = "light is blinking red"
predicted_intent = predict_intent(user_input)
print(f"Predicted Intent: {predicted_intent}")
