"""
db_memory.py — Memory databases including session (short-term) vector DB.
"""
import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

EMBEDDING_DIM = 3072

# ── Permanent DB (resume) ─────────────────────────────────────────
if os.path.exists("faiss_permanent"):
    permanent_db = FAISS.load_local("faiss_permanent", embeddings, allow_dangerous_deserialization=True)
else:
    permanent_db = None

# ── Long-Term DB (saved facts) ────────────────────────────────────
if os.path.exists("faiss_long_term"):
    long_term_db = FAISS.load_local("faiss_long_term", embeddings, allow_dangerous_deserialization=True)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    long_term_db = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )

# ── Session DB (short-term, in-memory, wiped on new chat) ─────────
def _new_session_db():
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )

session_db = _new_session_db()
session_doc_name = None   # tracks which document is loaded

def reset_session_db():
    """Wipe session DB — called on New Chat."""
    global session_db, session_doc_name
    session_db = _new_session_db()
    session_doc_name = None

def has_session_docs() -> bool:
    return session_db.index.ntotal > 0