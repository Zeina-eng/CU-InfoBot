import os
import streamlit as st
import base64

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from streamlit_option_menu import option_menu

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

/* Document Browser */

.document-browser {{

    background: rgba(255,255,255,0.98);

    padding: 25px;

    border-radius: 18px;

    margin-top: 20px;

    margin-bottom: 20px;

    box-shadow: 0 8px 20px rgba(0,0,0,0.35);

}}

/* Title */

.document-browser h2 {{

    color: #6A1B9A !important;

    font-size: 30px;

    font-weight: bold;

}}

/* Subtitle */

.document-browser p {{

    color: #222 !important;

    font-size: 17px;

    line-height: 1.7;

}}

/* Document cards */

.document-browser .chunk-box {{

    background: #ffffff;

    border: 1px solid #dddddd;

    border-radius: 12px;

    padding: 20px;

    margin-bottom: 20px;

}}

/* Card headings */

.document-browser .chunk-box h3 {{

    color: #6A1B9A !important;

    font-size: 22px;

    font-weight: bold;

}}

/* Card text */

.document-browser .chunk-box p {{

    color: #111 !important;

    font-size: 17px;

    line-height: 1.8;

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
# Load LLM
# ---------------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0
)

# -------------------------------------------------
# Sidebar - Document Browser (Independent)
# -------------------------------------------------

with st.sidebar:

    st.title("📚 University Documents")

    selected_category = option_menu(
        "Browse Documents",
        [
            "Leave Policy",
            "Hostel",
            "Examination Rules",
            "Academic Calendar",
            "Course Syllabus",
            "Notices",
        ],
        icons=[
            "file-earmark-text",
            "house",
            "clipboard-check",
            "calendar-event",
            "book",
            "megaphone"
        ],
        default_index=0
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

    with st.spinner("Searching university documents..."):

        try:

            # ---------------------------
            # Retrieve relevant chunks
            # ---------------------------
            question_docs = db.similarity_search(
                question,
                k=5
            )



            # ---------------------------
            # Build context
            # ---------------------------
            context = "\n\n".join(
                [doc.page_content for doc in question_docs]
            )

            # ---------------------------
            # Prompt
            # ---------------------------
            prompt = f"""
You are CU InfoBot, an AI assistant for Chandigarh University.

Instructions:

- Answer ONLY using the provided context.
- If multiple retrieved chunks are available, combine them into one complete answer.
- Prefer chunks that directly answer the user's question.
- Do not answer from general knowledge.
- If the answer is not found, reply exactly:

I could not find this information in the available university documents.

Context:
{context}

Question:
{question}

Answer:
"""       # ---------------------------
            # Generate answer
            # ---------------------------
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

<p>{answer}</p>

</div>
""",
                unsafe_allow_html=True,
            )

            # ---------------------------
            # Retrieved Chunks
            # ---------------------------
            with st.expander("📄 Retrieved Document Chunks", expanded=False):

                if len(question_docs) == 0:

                    st.warning("No relevant document chunks were found.")

                else:

                    for i, doc in enumerate(question_docs, start=1):

                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "N/A")
                        category = doc.metadata.get("category", "Unknown")

                        st.markdown(
                            f"""
<div class="chunk-box">

<h3>Chunk {i}</h3>

<p><b>Category:</b> {category}</p>

<p><b>Source:</b> {source}</p>

<p><b>Page:</b> {page}</p>

<hr>

<p>{doc.page_content}</p>

</div>
""",
                            unsafe_allow_html=True,
                        )

        except Exception as e:

            st.error(f"Error: {e}")
        
# -------------------------------------------------
# Show selected document (Independent of chatbot)
# -------------------------------------------------

st.markdown("---")

st.markdown(
    f"""
<div class="document-browser">

<h2>📄 {selected_category}</h2>

<p><b>Document Chunks</b></p>

""",
    unsafe_allow_html=True,
)

category_docs = db.similarity_search(
    selected_category,
    k=5
)

for i, doc in enumerate(category_docs, start=1):

    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "N/A")
    category = doc.metadata.get("category", "Unknown")

    st.markdown(
        f"""
<div class="chunk-box">

<h3>Chunk {i}</h3>

<p><b>Category:</b> {category}</p>

<p><b>Source:</b> {source}</p>

<p><b>Page:</b> {page}</p>

<hr>

<p>{doc.page_content}</p>

</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)

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