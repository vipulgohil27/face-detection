# jd_parser.py
from keybert import KeyBERT
import requests


def extract_keywords_from_jd_url(jd_url, num_keywords=10):
    #response = requests.get(jd_url)
    jd_text = open(jd_url).read()
    #jd_text = response.text
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(jd_text, top_n=num_keywords)
    print(keywords)
    return [kw[0] for kw in keywords]
