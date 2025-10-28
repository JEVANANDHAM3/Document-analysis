import streamlit as st
import PyPDF2
import docx
import os
import time
from numpy.random import default_rng as rng
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import textwrap
from concurrent.futures import ThreadPoolExecutor
from streamlit.elements.lib.layout_utils import Height
from streamlit_mermaid import st_mermaid
import streamlit.components.v1 as components
import re
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
import spacy

from utils import chunk_text, remove_mermaid_fences
from semantic_scoring import get_cosine_scores, model_load
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

@st.cache_data
def get_word_frequency(text: str, top_n: int = 20):
    """Compute word frequency from text."""
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 
                  'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                  'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    word_counts = Counter(filtered_words)
    return word_counts.most_common(top_n)

@st.cache_data
def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50):
    """Split document into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@st.cache_data
def get_embeddings(chunks: list[str]):
    """Get vector embeddings for chunks using spaCy."""
    nlp = model_load()
    embeddings = []
    for chunk in chunks:
        doc = nlp(chunk)
        embeddings.append(doc.vector)
    return np.array(embeddings)

@st.cache_data
def cluster_chunks(embeddings: np.ndarray, n_clusters: int):
    """Cluster embeddings using KMeans."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels

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

if uploaded_file is not None:
    st.subheader("📊 Frequency Analysis")
    if st.button("Analyze Word Frequency"):
        with st.spinner("Analyzing word frequency..."):
            word_freq = get_word_frequency(content, top_n=20)
            
            # Create DataFrame
            df_freq = pd.DataFrame(word_freq, columns=['Word', 'Count'])
            
            # Create bar chart
            fig = px.bar(
                df_freq,
                x='Word',
                y='Count',
                title='Top 20 Most Frequent Words',
                labels={'Word': 'Word', 'Count': 'Frequency'},
                color='Count',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(df_freq, use_container_width=True)
            
        st.success("Frequency analysis completed!")

if uploaded_file is not None:
    st.subheader("🔍 Document Clustering")
    
    # User input for number of clusters
    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
    
    if st.button("Cluster Document"):
        with st.spinner("Clustering document chunks..."):
            # Chunk the document
            chunks = chunk_document(content, chunk_size=500, overlap=50)
            
            if len(chunks) < n_clusters:
                st.error(f"Document has only {len(chunks)} chunks. Please reduce the number of clusters.")
            else:
                # Get embeddings
                embeddings = get_embeddings(chunks)
                
                # Cluster
                labels = cluster_chunks(embeddings, n_clusters)
                
                # Display results
                st.write(f"**Total chunks:** {len(chunks)}")
                st.write(f"**Number of clusters:** {n_clusters}")
                
                # Group chunks by cluster
                for cluster_id in range(n_clusters):
                    with st.expander(f"📁 Cluster {cluster_id + 1} ({sum(labels == cluster_id)} chunks)"):
                        cluster_chunks_list = [chunks[i] for i in range(len(chunks)) if labels[i] == cluster_id]
                        
                        for idx, chunk in enumerate(cluster_chunks_list):
                            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                            st.markdown(f"**Chunk {idx + 1}:**")
                            st.text(preview)
                            st.markdown("---")
                
                # Cluster distribution chart
                cluster_counts = pd.DataFrame({
                    'Cluster': [f'Cluster {i+1}' for i in range(n_clusters)],
                    'Count': [sum(labels == i) for i in range(n_clusters)]
                })
                
                fig_cluster = px.pie(
                    cluster_counts,
                    values='Count',
                    names='Cluster',
                    title='Cluster Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                st.plotly_chart(fig_cluster, use_container_width=True)
                
        st.success("Clustering completed!")

