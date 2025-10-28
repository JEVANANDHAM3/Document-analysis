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
from sklearn.decomposition import PCA
import spacy

from utils import chunk_text, remove_mermaid_fences
from semantic_scoring import get_cosine_scores, model_load
from mermaid import generate_mermaid

# Page configuration
st.set_page_config(
    page_title="Document Analysis Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    .analysis-section {
        background: #f7fafc;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


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
    stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
                'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
                'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn',
                'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'must', 'may', 'one',
                'would', 'could', 'might', 'time', 'like', 'get'}
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
    return labels, kmeans

@st.cache_data
def reduce_dimensions(embeddings: np.ndarray, n_components: int = 2):
    """Reduce embeddings to 2D using PCA for visualization."""
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca

# Main header
st.markdown('<h1 class="main-header">📄 Document Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-Powered Document Intelligence Platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/document.png", width=80)
    st.title("Navigation")
    st.markdown("---")
    st.info("📌 **Upload a document** to unlock all analysis features")
    
    st.markdown("### Features")
    st.markdown("""
    - 📊 Sentiment Analysis
    - 🧠 Mind Map Generation
    - 📈 Frequency Analysis
    - 🔍 Document Clustering
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This dashboard uses advanced NLP and ML techniques to analyze your documents.")

uploaded_file = st.file_uploader(
    "📁 Choose a document (TXT, PDF, or DOCX)", 
    type=["txt", "pdf", "docx"],
    help="Upload a text, PDF, or Word document for analysis"
)

if uploaded_file is not None:

    with st.spinner("🔄 Processing document..."):
        content = ""
        file_extension = os.path.splitext(uploaded_file.name)[1]

        if file_extension == ".txt":
            content = read_txt(uploaded_file)
        elif file_extension == ".pdf":
            content = read_pdf(uploaded_file)
        elif file_extension == ".docx":
            content = read_docx(uploaded_file)
    
    # Document metrics
    col1, col2, col3, col4 = st.columns(4)
    
    word_count = len(content.split())
    char_count = len(content)
    sentence_count = len(re.split(r'[.!?]+', content))
    
    with col1:
        st.metric("📝 Words", f"{word_count:,}")
    with col2:
        st.metric("🔤 Characters", f"{char_count:,}")
    with col3:
        st.metric("📄 Sentences", f"{sentence_count:,}")
    with col4:
        st.metric("📂 File Type", file_extension.upper())
    
    st.success("✅ Document loaded successfully!")

    with st.expander("📖 View Document Content"):
        st.text_area("Content", content, height=400, label_visibility="collapsed")
        
    st.markdown("---")
else:
    st.warning("⚠️ Please upload a document to get started.")
    st.stop()


# Analysis Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment Analysis", "🧠 Mind Map", "📈 Frequency Analysis", "🔍 Clustering"])

with tab1:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.subheader("📊 Multi-Trend Sentiment Analysis")
    st.markdown("Analyze sentiment trends across different categories in your document.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        categories = st.text_input(
            "Enter sentiment categories (comma-separated)",
            placeholder="e.g., positive, negative, neutral, excited, concerned",
            help="Enter multiple sentiment categories to analyze"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze Sentiment", key="sentiment_btn")
    
    if analyze_btn and categories:
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
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Categories Analyzed", len(categories_list))
        with col2:
            st.metric("Document Chunks", len(chunk_text))
        with col3:
            avg_score = df_melted['score'].mean()
            st.metric("Average Score", f"{avg_score:.2f}")

        st.success("✅ Sentiment analysis completed!")
    elif analyze_btn:
        st.error("Please enter sentiment categories.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.subheader("🧠 AI-Powered Mind Map Generation")
    st.markdown("Generate a visual mind map to understand the key concepts and relationships in your document.")
    
    if st.button("🎨 Generate Mind Map", key="mindmap_btn"):
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
        st.success("✅ Mind map generated successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.subheader("📈 Word Frequency Analysis")
    st.markdown("Discover the most frequently used words in your document.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        top_n = st.number_input("Number of words", min_value=5, max_value=50, value=20, step=5)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        freq_btn = st.button("📊 Analyze Frequency", key="freq_btn")
    
    if freq_btn:
        with st.spinner("🔄 Analyzing word frequency..."):
            word_freq = get_word_frequency(content, top_n=top_n)
            
            # Create DataFrame
            df_freq = pd.DataFrame(word_freq, columns=['Word', 'Count'])
            
            # Create bar chart
            # Create enhanced bar chart
            fig = px.bar(
                df_freq,
                x='Word',
                y='Count',
                title=f'Top {top_n} Most Frequent Words',
                labels={'Word': 'Word', 'Count': 'Frequency'},
                color='Count',
                color_continuous_scale='Viridis',
                text='Count'
            )
            
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(
                xaxis_tickangle=-45,
                height=600,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table with styling
            st.markdown("### 📋 Detailed Word Frequency Table")
            st.dataframe(
                df_freq.style.background_gradient(cmap='Blues', subset=['Count']),
                use_container_width=True,
                height=400
            )
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Unique Words", len(word_freq))
            with col2:
                st.metric("Most Frequent Word", df_freq.iloc[0]['Word'])
            with col3:
                st.metric("Highest Frequency", df_freq.iloc[0]['Count'])
            
        st.success("✅ Frequency analysis completed!")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
    st.subheader("🔍 Advanced Document Clustering")
    st.markdown("Cluster document chunks to discover thematic patterns and group similar content.")
    
    # User input for clustering parameters
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3, help="Select how many clusters to create")
    with col2:
        chunk_size = st.slider("Chunk size (characters)", min_value=200, max_value=1000, value=500, step=100)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        cluster_btn = st.button("🎯 Cluster Document", key="cluster_btn")
    
    if cluster_btn:
        with st.spinner("🔄 Clustering document chunks..."):
            # Chunk the document
            chunks = chunk_document(content, chunk_size=chunk_size, overlap=50)
            
            if len(chunks) < n_clusters:
                st.error(f"Document has only {len(chunks)} chunks. Please reduce the number of clusters.")
            else:
                # Get embeddings
                embeddings = get_embeddings(chunks)
                
                # Cluster
                labels, kmeans = cluster_chunks(embeddings, n_clusters)
                
                # Reduce dimensions for visualization
                reduced_embeddings, pca = reduce_dimensions(embeddings, n_components=2)
                
                # Create visualization dataframe
                viz_df = pd.DataFrame({
                    'x': reduced_embeddings[:, 0],
                    'y': reduced_embeddings[:, 1],
                    'Cluster': [f'Cluster {i+1}' for i in labels],
                    'Chunk_ID': [f'Chunk {i+1}' for i in range(len(chunks))],
                    'Preview': [chunk[:100] + '...' if len(chunk) > 100 else chunk for chunk in chunks]
                })
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📦 Total Chunks", len(chunks))
                with col2:
                    st.metric("🎯 Clusters", n_clusters)
                with col3:
                    st.metric("📊 Variance Explained", f"{sum(pca.explained_variance_ratio_)*100:.1f}%")
                with col4:
                    avg_cluster_size = len(chunks) / n_clusters
                    st.metric("📈 Avg Cluster Size", f"{avg_cluster_size:.1f}")
                
                st.markdown("---")
                
                # Create scatter plot
                st.markdown("### 🎨 Cluster Visualization (2D PCA Projection)")
                fig_scatter = px.scatter(
                    viz_df,
                    x='x',
                    y='y',
                    color='Cluster',
                    hover_data=['Chunk_ID', 'Preview'],
                    title='Document Chunks Clustered in 2D Space',
                    labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    size_max=15
                )
                
                fig_scatter.update_traces(
                    marker=dict(size=12, line=dict(width=2, color='white')),
                    selector=dict(mode='markers')
                )
                
                fig_scatter.update_layout(
                    height=600,
                    plot_bgcolor='rgba(240,240,240,0.5)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.markdown("---")
                
                # Cluster distribution charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    cluster_counts = pd.DataFrame({
                        'Cluster': [f'Cluster {i+1}' for i in range(n_clusters)],
                        'Count': [sum(labels == i) for i in range(n_clusters)]
                    })
                    
                    fig_pie = px.pie(
                        cluster_counts,
                        values='Count',
                        names='Cluster',
                        title='Cluster Distribution (Pie Chart)',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        hole=0.4
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_bar = px.bar(
                        cluster_counts,
                        x='Cluster',
                        y='Count',
                        title='Cluster Distribution (Bar Chart)',
                        color='Count',
                        color_continuous_scale='Viridis',
                        text='Count'
                    )
                    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                    fig_bar.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("---")
                
                # Group chunks by cluster with enhanced UI
                st.markdown("### 📂 Cluster Details")
                for cluster_id in range(n_clusters):
                    cluster_size = sum(labels == cluster_id)
                    with st.expander(f"📁 Cluster {cluster_id + 1} - {cluster_size} chunks ({cluster_size/len(chunks)*100:.1f}%)", expanded=False):
                        cluster_chunks_list = [chunks[i] for i in range(len(chunks)) if labels[i] == cluster_id]
                        
                        for idx, chunk in enumerate(cluster_chunks_list):
                            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                            st.markdown(f"**📄 Chunk {idx + 1}** ({len(chunk)} characters)")
                            st.info(preview)
                            if idx < len(cluster_chunks_list) - 1:
                                st.markdown("---")
                
        st.success("✅ Clustering analysis completed!")
    
    st.markdown('</div>', unsafe_allow_html=True)

