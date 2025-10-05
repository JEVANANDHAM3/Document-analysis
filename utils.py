from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def chunk_text(text:str,chunk_size:int = 10, chunk_overlap:int = 3) -> list[str]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n"," ", ""]
    )

    chunks = text_splitter.split_text(text)

    return chunks


def remove_mermaid_fences(text: str) -> str:
    """
    Remove the ```mermaid fences from a string, keeping the content inside.
    """
    # Remove ```mermaid at the start
    text = re.sub(r'```mermaid', '', text)
    # Remove closing ```
    text = re.sub(r'```', '', text)
    # Strip extra whitespace
    return text.strip()
