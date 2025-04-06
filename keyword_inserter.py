# keyword_inserter.py
def add_keywords_to_resume(resume_text, keywords):
    missing = [kw for kw in keywords if kw.lower() not in resume_text.lower()]
    if not missing:
        return resume_text
    return resume_text + "\n\nSkills & Keywords:\n" + ", ".join(missing)

