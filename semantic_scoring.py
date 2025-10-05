import spacy
import streamlit as st

@st.cache_resource
def model_load():
    nlp = spacy.load("en_core_web_lg")  
    return nlp

@st.cache_data
def get_cosine_scores(texts: list[str], categories: list[str]) -> list[dict]:
    nlp = model_load()
    category_docs = [nlp(c) for c in categories]

    results = []
    for text in texts:
        t_doc = nlp(text)
        sims = [t_doc.similarity(c_doc) for c_doc in category_docs]
        total = sum(sims) or 1  
        probs = [round(s / total, 2) for s in sims]
        results.append({"text": text, **dict(zip(categories, probs))})

    return results  
