import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(
    page_title="CU InfoBot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 CU InfoBot")

st.write(
    "Your AI-powered assistant for accessing accurate information from Chandigarh University documents, policies, and academic resources."
)

st.info(
    "Ask questions related to uploaded Chandigarh University documents such as leave policy, hostel policy, academic calendar, examination schedules, and course information."
)

# ---------------------------
# Check vector DB exists
# ---------------------------
if not os.path.exists("vector_db"):
    st.error("❌ Vector DB not found. Please create embeddings first.")
    st.stop()

# ---------------------------
# Load embeddings model
# ---------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# Load FAISS database
# ---------------------------
db = FAISS.load_local(
    "vector_db",
    embedding,
    allow_dangerous_deserialization=True
)

# ---------------------------
# Load Ollama model
# ---------------------------
llm = OllamaLLM(
    model="qwen2.5:1.5b"
)

# ---------------------------
# User input
# ---------------------------
question = st.text_input(
    "Ask a Chandigarh University Question"
)

# ---------------------------
# Process question
# ---------------------------
if question:

    with st.spinner("Searching documents..."):

        try:
            # Retrieve relevant chunks
            docs = db.similarity_search(question, k=5)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
You are CU InfoBot, an assistant for Chandigarh University.

Rules:
1. Answer ONLY from the provided context.
2. Do NOT make up information.
3. If the answer is not clearly present in the context, reply exactly:

I could not find this information in the available university documents.

Context:
{context}

Question:
{question}

Answer:
"""

            answer = llm.invoke(prompt)

            st.subheader("Answer")
            st.success(answer)

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ---------------------------
    # Show retrieved chunks
    # ---------------------------
    with st.expander("📄 Retrieved Document Chunks"):

        for i, doc in enumerate(docs, start=1):

            st.markdown(f"### Chunk {i}")

            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            st.write(f"📄 Source: {source}")
            st.write(f"📑 Page: {page}")

            st.write(doc.page_content[:1000])

            st.markdown("---")