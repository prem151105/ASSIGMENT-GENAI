"""
ingest.py
---------
Step 1-4 of the assignment:
  1. Load your PDF/notes
  2. Split text into chunks (RecursiveCharacterTextSplitter)
  3. Embed chunks (HuggingFaceEmbeddings - free, runs locally)
  4. Store embeddings in a FAISS vector index on disk

Usage:
    python ingest.py

It will read every .pdf and .txt file inside the `data/` folder and build
a FAISS index inside `faiss_index/`.
"""

import os
import glob

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    """Load all PDF and TXT files from the data/ directory."""
    documents = []

    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))

    if not pdf_files and not txt_files:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in '{DATA_DIR}/'. "
            "Add your notes there before running ingest.py"
        )

    for path in pdf_files:
        print(f"Loading PDF: {path}")
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    for path in txt_files:
        print(f"Loading TXT: {path}")
        loader = TextLoader(path, encoding="utf-8")
        documents.extend(loader.load())

    return documents


def build_index():
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s)/page(s).")

    # Step 3: Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Step 4: Embed chunks (free, local HuggingFace model)
    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Build FAISS vector store
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"FAISS index saved to '{INDEX_DIR}/'.")


if __name__ == "__main__":
    build_index()
