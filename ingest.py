"""
ingest.py — Build the knowledge base (RAG index) from the Constitution of India PDF.

Workflow step from the architecture diagram: "Searches Knowledge Base (Vector DB)".
Runs once at BUILD time on Render (see render.yaml), so the index ships with the app.

Free stack:
  - pypdf                  -> read the PDF
  - RecursiveCharacterTextSplitter -> chunk it
  - FastEmbed (ONNX)       -> free local embeddings, light enough for Render's 512MB free tier
  - ChromaDB               -> free, embedded vector database (no server needed)
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

PDF_PATH = os.getenv("PDF_PATH", "data/constitution.pdf")
DB_DIR = os.getenv("DB_DIR", "chroma_db")


def build_index() -> None:
    if os.path.isdir(DB_DIR) and os.listdir(DB_DIR):
        print(f"[ingest] Index already exists at {DB_DIR}, skipping.")
        return

    print(f"[ingest] Loading PDF: {PDF_PATH}")
    docs = PyPDFLoader(PDF_PATH).load()
    print(f"[ingest] Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Split into {len(chunks)} chunks")

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print("[ingest] Building Chroma index (this takes a few minutes on free CPU)...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="constitution_of_india",
    )
    print(f"[ingest] Done. Index saved to {DB_DIR}")


if __name__ == "__main__":
    build_index()
