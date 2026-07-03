import os
import streamlit as st
import base64

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

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
Your AI-powered assistant for accessing accurate information from Chandigarh University documents, policies, and academic resources."
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

    st.markdown("""
    <style>

    /* Sidebar */
    section[data-testid="stSidebar"]{
        background:#ffffff;
        border-right:1px solid #d9d9d9;
    }

    /* Logo */
    .logo{
        font-size:30px;
        font-weight:bold;
        color:#162a72;
        padding:15px 10px;
    }

    /* Menu Items */
    .menu-item{
        padding:14px 18px;
        font-size:17px;
        border-bottom:1px solid #f1f1f1;
        cursor:pointer;
        transition:0.2s;
    }

    .menu-item:hover{
        background:#f2f4fa;
    }

    /* Active item */
    .active{
        background:#162a72;
        color:white;
        font-weight:bold;
        border-radius:3px;
    }

    .arrow{
        float:right;
        font-weight:bold;
    }

    .bottom-box{
        margin-top:30px;
        padding:15px;
        border:1px solid #e5e5e5;
        border-radius:10px;
        background:white;
    }

    </style>

    <div class="logo">
    🎓 CU InfoBot
    </div>

    <div class="menu-item">
    Academics <span class="arrow">›</span>
    </div>

    <div class="menu-item active">
    Accounts <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Administration <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Admission Document Upload
    </div>

    <div class="menu-item">
    Apply for Loan Documents
    </div>

    <div class="menu-item">
    Apply for NOC
    </div>

    <div class="menu-item">
    Centre For Student Wellbeing <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Counseling Therapy Clinic
    </div>

    <div class="menu-item">
    DCPD <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    E Library <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Examination <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Hostel <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Placements <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Scholarships <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Student Services <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Transport <span class="arrow">›</span>
    </div>

    <div class="menu-item">
    Other Important Links <span class="arrow">›</span>
    </div>

    <div class="bottom-box">
    <b>🤖 CU InfoBot</b><br>
    AI-Powered Assistant for Chandigarh University
    </div>

    """, unsafe_allow_html=True)
    
# ---------------------------
# Check vector DB exists
# ---------------------------
if not os.path.exists("vector_db"):
    st.error("❌ Vector DB not found. Please create embeddings first.")
    st.stop()

# ---------------------------
# Load embeddings model
# ---------------------------
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding = load_embedding()
# ---------------------------
# Load FAISS database
# ---------------------------
@st.cache_resource
def load_db():
    return FAISS.load_local(
        "vector_db",
        embedding,
        allow_dangerous_deserialization=True
    )

db = load_db()

# ---------------------------
# Load Ollama model
# ---------------------------
import streamlit as st

llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature=0
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
            docs = db.similarity_search(
                question.lower(),
                k=5
            )

            # Build context
            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            # Prompt
            prompt = f"""
You are CU InfoBot, an AI assistant for Chandigarh University.

Instructions:

- Answer ONLY from the provided context.
- Do NOT make up information.
- If information is unavailable, reply exactly:

I could not find this information in the available university documents.

- For attendance questions:
Provide attendance percentage requirements.

- For examination questions:
Provide examination schedules and rules.

- For syllabus questions:
Provide subject names and syllabus details.

- For academic calendar questions:
Provide dates and important events.

- For hostel questions:
Provide hostel policies and regulations.

Context:
{context}

Question:
{question}

Answer:
"""

            # Generate answer
            response = llm.invoke(prompt)
            answer = response.content

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
                unsafe_allow_html=True,
            )

            # ---------------------------
            # Retrieved Chunks
            # ---------------------------

            with st.expander("📄 Retrieved Document Chunks"):

                for i, doc in enumerate(docs, start=1):

                    source = doc.metadata.get("source", "Unknown")
                    category = doc.metadata.get("category", "Unknown")
                    page = doc.metadata.get("page", "N/A")

                    st.markdown(
                        f"""
<div class="chunk-box">

<h3>Chunk {i}</h3>

<p><b>📂 Category:</b> {category}</p>

<p><b>📄 Source:</b> {source}</p>

<p><b>📑 Page:</b> {page}</p>

<hr>

<p>{doc.page_content[:1000]}</p>

</div>
""",
                        unsafe_allow_html=True,
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
Built with Streamlit, LangChain, FAISS and Groq

</div>
""",
unsafe_allow_html=True
)