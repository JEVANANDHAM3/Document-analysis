from google import genai
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

gemini_api_key:str = str(os.getenv("GEMINI_API_KEY"))
@st.cache_data
def generate_mermaid(content:str,api_key:str=gemini_api_key):
    prompt = f"""
    You are an expert at creating concise and visually balanced mind maps from raw text.

    Task:
    - Read the following content carefully.
    - Identify **only the 3–4 most important main branches (core ideas)**.
    - For each main branch, include **up to 2–3 subtopics only**.
    - Keep the hierarchy **no deeper than 4 levels** in total.
    - Ensure the final mind map is **compact, symmetrical, and easy to read**.
    - Generate a **valid Mermaid.js mind map** using **left-to-right layout (`graph LR`)**.
    - Replace spaces in node names with underscores.
    - Avoid duplicate edges, long text nodes, or redundant relationships.
    - Output **only the Mermaid code**, without any extra text or explanations.

    Content:
    \"\"\"{content}\"\"\"

    Output:
    """



    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text