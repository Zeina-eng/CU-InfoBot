import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------------------------
# Load all PDFs
# ---------------------------

all_docs = []

documents_folder = "documents"

for root, dirs, files in os.walk(documents_folder):

    for file in files:

        if file.lower().endswith(".pdf"):

            pdf_path = os.path.join(root, file)

            print(f"Loading: {pdf_path}")

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            # Add metadata
            category = os.path.basename(root)

            for doc in docs:

                doc.metadata["source"] = file

                doc.metadata["category"] = category

            all_docs.extend(docs)

print(f"\nTotal Pages Loaded: {len(all_docs)}")


# ---------------------------
# Split documents
# ---------------------------

splitter = RecursiveCharacterTextSplitter(

    chunk_size=700,

    chunk_overlap=150

)

chunks = splitter.split_documents(all_docs)

print(f"Total Chunks Created: {len(chunks)}")


# ---------------------------
# Embedding model
# ---------------------------

embedding = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)

print("\nCreating embeddings and FAISS index...")


# ---------------------------
# Delete old vector_db
# ---------------------------

if os.path.exists("vector_db"):

    import shutil

    shutil.rmtree("vector_db")


# ---------------------------
# Create FAISS database
# ---------------------------

db = FAISS.from_documents(

    chunks,

    embedding

)


# ---------------------------
# Save database
# ---------------------------

db.save_local("vector_db")

print("\n✅ Vector Database Created Successfully!")

print("📁 Saved in: vector_db/")