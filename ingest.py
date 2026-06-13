import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

all_docs = []

# ---------------------------
# Load PDFs from documents folder
# ---------------------------
for root, dirs, files in os.walk("documents"):
    for file in files:
        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(root, file)

            print(f"Loading: {pdf_path}")

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            all_docs.extend(docs)

print(f"\nTotal Pages Loaded: {len(all_docs)}")

# ---------------------------
# Split into chunks
# ---------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

print(f"Total Chunks Created: {len(chunks)}")

# ---------------------------
# Embeddings
# ---------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nCreating embeddings and FAISS index...")

# ---------------------------
# Create FAISS DB
# ---------------------------
db = FAISS.from_documents(
    chunks,
    embedding
)

# ---------------------------
# Save DB
# ---------------------------
db.save_local("vector_db")

print("\n✅ Vector Database Created Successfully!")
print("📁 Saved in: vector_db/")