import os
import streamlit as st
import base64

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="CU InfoBot",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------
# BACKGROUND IMAGE
# ---------------------------
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()

    return base64.b64encode(data).decode()


bg_image = get_base64(
    "images/chandigarh-univ-165581419816x9.avif"
)


page_bg = f"""
<style>

/* Background */

.stApp {{
    background-image: url("data:image/avif;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* Header */

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

/* Main container */

.main-container {{
    background: rgba(0,0,0,0.55);
    padding: 25px;
    border-radius: 20px;
}}

/* Main white text */

.main-container h1,
.main-container h2,
.main-container h3,
.main-container h4,
.main-container p,
label {{

    color: white !important;
}}

/* Answer box */

.answer-box {{

    background-color: rgba(255,255,255,0.95);

    color: black !important;

    padding: 20px;

    border-radius: 15px;

    font-size: 18px;

    box-shadow: 0 0 15px rgba(0,0,0,0.3);
}}

.answer-box * {{

    color: black !important;
}}

/* Retrieved chunks */

.chunk-box {{

    background: white;

    color: black !important;

    padding: 20px;

    border-radius: 15px;

    margin-bottom: 20px;
}}

.chunk-box * {{

    color: black !important;
}}

/* Sidebar */

[data-testid="stSidebar"] {{

    background: rgba(255,255,255,0.95);
}}

[data-testid="stSidebar"] * {{

    color: black !important;
}}

[data-testid="stSidebar"] h1,

[data-testid="stSidebar"] h2,

[data-testid="stSidebar"] h3 {{

    color: #6A1B9A !important;

    font-weight: bold;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)
# ---------------------------
# HEADER
# ---------------------------

st.markdown(
"""
<div class="main-container">

<h1 style='text-align:center;'>
🎓 CU InfoBot
</h1>

<h3 style='text-align:center;'>
AI-Powered Chandigarh University Assistant
</h3>

<p style='text-align:center;'>
Ask questions about leave policy, hostel rules,
academic calendar, examination schedules,
syllabus and university regulations.
</p>

</div>
""",
unsafe_allow_html=True
)

st.write("")
# ---------------------------
# SIDEBAR
# ---------------------------

with st.sidebar:

    

    st.title("🎓 CU InfoBot")

    st.markdown("---")

    st.markdown("""
### Features
✅ University Leave Policy

✅ Academic Calendar

✅ Examination Guidelines

✅ Hostel Rules

✅ University Notices

✅ Course Information
""")

    st.markdown("---")

    st.info(
        "Your AI-powered assistant for accessing accurate information from Chandigarh University documents, policies, and academic resources."
    )

    st.markdown("---")

    st.caption("Powered by LangChain + FAISS + Ollama")

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

            # ---------------------------
            # Show Answer
            # ---------------------------

            st.subheader("Answer")

            st.markdown(
f"""
<div class="answer-box">

<h4>Answer</h4>

{answer}

</div>
""",
unsafe_allow_html=True
            )

            # ---------------------------
            # Show retrieved chunks
            # ---------------------------

            with st.expander("📄 Retrieved Document Chunks"):

                for i, doc in enumerate(docs, start=1):

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    category = doc.metadata.get(
                        "category",
                        "Unknown"
                    )

                    page = doc.metadata.get(
                        "page",
                        "N/A"
                    )

                    st.markdown(
f"""
<div style="
background:white;
padding:20px;
border-radius:15px;
margin-bottom:20px;
color:black !important;">

<h3 style="color:black;">Chunk {i}</h3>

<p style="color:black;">
<b>📂 Category:</b> {category}
</p>

<p style="color:black;">
<b>📄 Source:</b> {source}
</p>

<p style="color:black;">
<b>📑 Page:</b> {page}
</p>

<hr>

<p style="color:black;">
{doc.page_content[:1000]}
</p>

</div>
""",
unsafe_allow_html=True
)

        except Exception as e:

            st.error(f"Error: {e}")

            st.stop()

# ---------------------------
# FOOTER
# ---------------------------

st.markdown(
"""
<div style='text-align:center;
background:rgba(0,0,0,0.6);
padding:10px;
border-radius:10px;
color:white;
margin-top:20px;'>

CU InfoBot © 2026<br>
Built with Streamlit, LangChain, FAISS and Ollama

</div>
""",
unsafe_allow_html=True
)