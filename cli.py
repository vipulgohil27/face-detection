### resume-keyword-booster: Enhance resumes by injecting job-specific keywords

# jd_parser.py
# from keybert import KeyBERT
# import requests
#
#
# def extract_keywords_from_jd_url(jd_url, num_keywords=10):
#     response = requests.get(jd_url)
#     jd_text = response.text
#     kw_model = KeyBERT()
#     keywords = kw_model.extract_keywords(jd_text, top_n=num_keywords)
#     return [kw[0] for kw in keywords]


# # resume_reader.py
# import docx
#
#
# def load_resume_docx(path):
#     doc = docx.Document(path)
#     return "\n".join([para.text for para in doc.paragraphs])
#
#
# def save_resume_docx(text, path):
#     doc = docx.Document()
#     for line in text.split("\n"):
#         doc.add_paragraph(line)
#     doc.save(path)


# keyword_inserter.py
# def add_keywords_to_resume(resume_text, keywords):
#     missing = [kw for kw in keywords if kw.lower() not in resume_text.lower()]
#     if not missing:
#         return resume_text
#     return resume_text + "\n\nSkills & Keywords:\n" + ", ".join(missing)


# cli.py
import argparse
from jd_parser import extract_keywords_from_jd_url
from resume_reader import load_resume_docx, save_resume_docx
from keyword_inserter import add_keywords_to_resume


def main():
    parser = argparse.ArgumentParser(description="Boost your resume with JD keywords from URL!")
    parser.add_argument("--jd_url", required=True, help="URL to job description text")
    parser.add_argument("--resume", required=True, help="Path to resume .docx file")
    parser.add_argument("--out", required=True, help="Output path for updated resume")
    parser.add_argument("--top", type=int, default=10, help="Number of top keywords to extract")
    args = parser.parse_args()

    resume_text = load_resume_docx(args.resume)

    keywords = extract_keywords_from_jd_url(args.jd_url, args.top)

    updated_resume = add_keywords_to_resume(resume_text, keywords)

    save_resume_docx(updated_resume, args.out)
    print(f"✅ Resume updated and saved to {args.out}")


if __name__ == "__main__":
    main()

# requirements.txt
# keybert
# sentence - transformers
# docx
# python - docx
# requests
