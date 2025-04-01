from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

# Define request model
class AcronymRequest(BaseModel):
    sentence: str

# Load acronyms from CSV file
def load_acronyms(file_path="acronyms.csv"):
    df = pd.read_csv(file_path)
    acronym_dict = dict(zip(df["Acronym"].str.upper(), df["Meaning"]))
    return acronym_dict

ACRONYM_DICT = load_acronyms()

@app.post("/expand_acronyms/")
def expand_acronyms(request: AcronymRequest):
    """
    Replace acronyms/keywords in the given sentence with their meanings.
    """
    words = request.sentence.split()
    expanded_words = [ACRONYM_DICT.get(word.upper(), word) for word in words]
    expanded_sentence = " ".join(expanded_words)

    return {"original_sentence": request.sentence, "expanded_sentence": expanded_sentence}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)