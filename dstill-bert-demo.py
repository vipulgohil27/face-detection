#pip install transformer torch scipy
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from scipy.spatial.distance import cosine

#load distil bert tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_emedding(text):
    """convert text to embedding """
    inputs= tokenizer(text,return_tensors="pt",truncation=True,padding=True,max_length=512)
    with torch.no_grad():
        outputs=model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
def cosine_similarity(embedding1,embedding2):
    """calcuate cosine similarity between two embedding"""
    return 1-cosine(embedding1,embedding2)
statments= [
    "my internet is not working",
    "i have no net connection",
    "my wifi is down and nothing is loading",
    "my connection is slow but still works",
    "My modem is showing no signal",
    "give me pizza",
    "internet sucks"
]

embeddings=[get_emedding(text) for text in statments]

#compare sim between statements

print("\n pairwise sim scores")
for i in range(len(statments)):
    for j in range(i+1,len(statments)):
        sim=cosine_similarity(embeddings[i],embeddings[j])
        print(f"smil between:\n '{statments[i]}\n '{statments[j]}\n -> score: {sim}:.4f\n")