from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import sqlite3
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
if not os.path.exists("student.db"):
    from setup_db import create_database
    create_database()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PATH = "student.db"
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "sql_examples"

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# DATABASE SCHEMA (used in prompt)
# ─────────────────────────────────────────────
SCHEMA = """
Table: STUDENT
Columns:
  - NAME    (TEXT)  : Full name of the student
  - CLASS   (TEXT)  : Subject/class the student is enrolled in (e.g., Data Science, Math, Physics)
  - SECTION (TEXT)  : Section letter (e.g., A, B, C)
"""

# ─────────────────────────────────────────────
# EXAMPLE Q&A PAIRS — Stored in Vector DB
# ─────────────────────────────────────────────
EXAMPLE_PAIRS = [
    {"question": "How many students are there?",                        "sql": "SELECT COUNT(*) FROM STUDENT;"},
    {"question": "Show all students",                                   "sql": "SELECT * FROM STUDENT;"},
    {"question": "List students in Data Science class",                 "sql": 'SELECT * FROM STUDENT WHERE CLASS = "Data Science";'},
    {"question": "How many students are in section A?",                 "sql": 'SELECT COUNT(*) FROM STUDENT WHERE SECTION = "A";'},
    {"question": "Show all students in section B",                      "sql": 'SELECT * FROM STUDENT WHERE SECTION = "B";'},
    {"question": "What are the unique classes available?",              "sql": "SELECT DISTINCT CLASS FROM STUDENT;"},
    {"question": "Count students in each class",                        "sql": "SELECT CLASS, COUNT(*) FROM STUDENT GROUP BY CLASS;"},
    {"question": "Count students in each section",                      "sql": "SELECT SECTION, COUNT(*) FROM STUDENT GROUP BY SECTION;"},
    {"question": "Show students whose name starts with A",              "sql": "SELECT * FROM STUDENT WHERE NAME LIKE 'A%';"},
    {"question": "List all students sorted by name",                    "sql": "SELECT * FROM STUDENT ORDER BY NAME;"},
    {"question": "Show students in Math class section A",               "sql": 'SELECT * FROM STUDENT WHERE CLASS = "Math" AND SECTION = "A";'},
    {"question": "How many sections are there?",                        "sql": "SELECT COUNT(DISTINCT SECTION) FROM STUDENT;"},
    {"question": "Show top 5 students by name alphabetically",          "sql": "SELECT * FROM STUDENT ORDER BY NAME LIMIT 5;"},
    {"question": "Which class has the most students?",                  "sql": "SELECT CLASS, COUNT(*) AS total FROM STUDENT GROUP BY CLASS ORDER BY total DESC LIMIT 1;"},
    {"question": "Show students not in Data Science",                   "sql": 'SELECT * FROM STUDENT WHERE CLASS != "Data Science";'},
]

# ─────────────────────────────────────────────
# VECTOR STORE SETUP (ChromaDB + local embeddings)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_vector_store():
    """Initialize ChromaDB and populate with example Q&A pairs."""
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection if schema changed
    try:
        existing = chroma_client.get_collection(COLLECTION_NAME)
        if existing.count() == len(EXAMPLE_PAIRS):
            return existing  # Already populated, reuse
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        documents=[ex["question"] for ex in EXAMPLE_PAIRS],
        metadatas=[{"sql": ex["sql"]} for ex in EXAMPLE_PAIRS],
        ids=[f"ex_{i}" for i in range(len(EXAMPLE_PAIRS))]
    )
    return collection

# ─────────────────────────────────────────────
# RAG: Retrieve top-k similar examples
# ─────────────────────────────────────────────
def retrieve_examples(collection, question: str, top_k: int = 3) -> list[dict]:
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    examples = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        examples.append({"question": doc, "sql": meta["sql"]})
    return examples

# ─────────────────────────────────────────────
# LLM: Generate SQL using retrieved context
# ─────────────────────────────────────────────
def generate_sql(question: str, retrieved_examples: list[dict]) -> str:
    few_shot = "\n".join([
        f"Q: {ex['question']}\nSQL: {ex['sql']}"
        for ex in retrieved_examples
    ])

    system_prompt = f"""You are an expert SQL generator. Given a natural language question, generate a valid SQLite SQL query.

DATABASE SCHEMA:
{SCHEMA}

SIMILAR EXAMPLES (use these as reference):
{few_shot}

RULES:
- Return ONLY the raw SQL query. No explanations, no markdown, no backticks.
- Do not include the word 'sql' in your output.
- Use double quotes for string values in WHERE clauses.
- The table name is STUDENT (all uppercase).
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ],
        temperature=0.1,  # Low temp for consistent SQL
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────
# SQLITE: Execute query
# ─────────────────────────────────────────────
def run_query(sql: str, db_path: str):
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Named columns
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        conn.close()
        return rows, columns, None
    except Exception as e:
        return [], [], str(e)

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Text-to-SQL",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .stTextInput > div > div > input {
        background-color: #1e1e2e;
        color: #cdd6f4;
        border: 1px solid #45475a;
        border-radius: 8px;
    }
    .retrieved-box {
        background: #1e1e2e;
        border-left: 3px solid #89b4fa;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 RAG-Powered Text-to-SQL")
st.caption("Ask questions in plain English → Get SQL + Results using Retrieval-Augmented Generation")

# Init vector store once
with st.spinner("Loading vector store..."):
    collection = init_vector_store()

st.success(f"✅ Vector store ready — {collection.count()} examples loaded", icon="🗄️")
st.divider()

# Input
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input(
        "Ask a question about students:",
        placeholder="e.g. How many students are in Data Science class?",
        label_visibility="collapsed"
    )
with col2:
    submit = st.button("🔍 Ask", use_container_width=True)

# Processing
if submit and question.strip():
    with st.spinner("Retrieving similar examples..."):
        retrieved = retrieve_examples(collection, question, top_k=3)

    with st.spinner("Generating SQL with Llama 3..."):
        sql_query = generate_sql(question, retrieved)

    rows, columns, error = run_query(sql_query, DB_PATH)

    # ── Show retrieved examples (the RAG part) ──
    with st.expander("📚 Retrieved Examples (RAG Context)", expanded=True):
        st.markdown("These similar examples were retrieved from the vector store and used as context:")
        for i, ex in enumerate(retrieved, 1):
            st.markdown(
                f'<div class="retrieved-box"><b>#{i} Q:</b> {ex["question"]}<br>'
                f'<b>SQL:</b> <code>{ex["sql"]}</code></div>',
                unsafe_allow_html=True
            )

    # ── Generated SQL ──
    st.subheader("🖊️ Generated SQL Query")
    st.code(sql_query, language="sql")

    # ── Results ──
    st.subheader("📊 Query Results")
    if error:
        st.error(f"SQL Error: {error}")
    elif not rows:
        st.info("No results found.")
    else:
        import pandas as pd
        df = pd.DataFrame([dict(zip(columns, row)) for row in rows])
        st.dataframe(df, use_container_width=True)
        st.caption(f"{len(rows)} row(s) returned")

elif submit and not question.strip():
    st.warning("Please enter a question first.")