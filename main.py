import streamlit as st
import PyPDF2
import docx
import os
import time
from numpy.random import default_rng as rng
import pandas as pd
import plotly.express as px
import textwrap
from concurrent.futures import ThreadPoolExecutor
from streamlit.elements.lib.layout_utils import Height
from streamlit_mermaid import st_mermaid
import streamlit.components.v1 as components

from utils import chunk_text, remove_mermaid_fences
from semantic_scoring import get_cosine_scores
from mermaid import generate_mermaid


@st.cache_data
def get_categories_array(text:str) -> list[str]:
    return [i.strip() for i in text.split(",")]

def wrap_text(text, width=60):
    return "<br>".join(textwrap.wrap(text, width=width))

def read_txt(file):
    """Reads and returns the content of a .txt file."""
    return file.getvalue().decode("utf-8")

def read_pdf(file):
    """Reads and returns the content of a .pdf file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    """Reads and returns the content of a .docx file."""
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

st.title("📄 Document Analysis Project")

uploaded_file = st.file_uploader(
    "Choose a document (TXT, PDF, or DOCX)", 
    type=["txt", "pdf", "docx"]
)

if uploaded_file is not None:

    with st.spinner("Processing document..."):
        content = ""
        file_extension = os.path.splitext(uploaded_file.name)[1]

        if file_extension == ".txt":
            content = read_txt(uploaded_file)
        elif file_extension == ".pdf":
            content = read_pdf(uploaded_file)
        elif file_extension == ".docx":
            content = read_docx(uploaded_file)
            

    st.success("Document loaded successfully!")

    with st.expander("Click to view the document's content"):
        st.text_area("Content", content, height=400)
        
    st.info("You can now use Multi-trend analysis,Mind map generation, Frequency analysis and Clustering document")
else:
    st.warning("Please upload a document to get started.")


if uploaded_file is not None:
    st.subheader("Sentiment Analysis")
    categories = st.text_area(label="positive,negative....",height=50)
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            categories_list = get_categories_array(categories)
            chunk_text = chunk_text(text=content,chunk_size=1000,chunk_overlap=100)
            
            sentiment_scores_list = get_cosine_scores(chunk_text, categories_list)


        df = pd.DataFrame(sentiment_scores_list)

        df['doc_num'] = range(1, len(df) + 1)


        df['text_wrapped'] = df['text'].apply(lambda t: wrap_text(t, width=60))

        df_melted = df.melt(
            id_vars=['doc_num', 'text_wrapped'],
            value_vars=categories_list,
            var_name='sentiment',
            value_name='score'
        )

        fig = px.line(
            df_melted, 
            x='doc_num',
            y='score',
            color='sentiment',
            markers=True,
            custom_data=['text_wrapped'],
        )

        fig.update_traces(
            hovertemplate=(
                "<b>Document #%{x}</b><br><br>" +
                "<b>Text</b>: %{customdata[0]}<br>" +
                "<b>Sentiment</b>: %{fullData.name}<br>" +
                "<b>Score</b>: %{y:.2f}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            title="Sentiment Score Trends with Detailed Hover Info",
            xaxis_title="Document Number",
            yaxis_title="Sentiment Score",
            legend_title="Sentiment"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.success("Chart is generated successfully!")

if uploaded_file is not None:
    st.subheader("Mind Map Generation")
    if st.button("Generate Mind Map"):
        with st.spinner("Generating Mind Map..."):
            mermaid = str(generate_mermaid(content=content))
            mermaid = remove_mermaid_fences(mermaid)
            print(mermaid)
            st.markdown(f"""
            <div style=" overflow: auto;">
                <script type="text/javascript">
                    {st_mermaid(mermaid)}  # pyright: ignore[reportArgumentType]
                </script>
            </div>
            """, unsafe_allow_html=True)
        st.success("Mind map is generated successfully!")

