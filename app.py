import os
import json
import tempfile
import requests
import streamlit as st

# ---------- Loaders & Splitters ----------
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader, TextLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- Embeddings ----------
from sentence_transformers import SentenceTransformer

# ---------- Qdrant ----------
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------- CONFIG ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CONTEXT_CHARS = 4000
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# ----------------------------------------


# ---------- Groq Streaming ----------
def groq_stream(messages, placeholder, model="llama-3.1-8b-instant"):

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        placeholder.error("❌ GROQ_API_KEY missing. Add it in Streamlit Cloud → Secrets.")
        return ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True
    }

    with requests.post(GROQ_API_URL, headers=headers, json=payload, stream=True) as r:
        r.raise_for_status()
        full = ""

        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            if line.strip() == "data: [DONE]":
                break

            try:
                chunk = json.loads(line.replace("data: ", ""))
                token = chunk["choices"][0]["delta"].get("content", "")
            except:
                token = ""

            full += token
            placeholder.markdown(full + "▌")

        placeholder.markdown(full)
        return full


# ---------- Prompts ----------
def prompt_qa(ctx, q):
    return [
        {"role": "system", "content": "Answer ONLY using the context. If the answer is not there, say 'I cannot answer from the text.'"},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q}\n\nAnswer:"}
    ]

def prompt_notes(ctx):
    return [
        {"role": "system", "content": "Create structured, clean notes."},
        {"role": "user", "content": f"Content:\n{ctx}\n\nWrite detailed notes:"}
    ]

def prompt_short(ctx):
    return [
        {"role": "system", "content": "Summarize concisely."},
        {"role": "user", "content": f"Content:\n{ctx}\n\nWrite 5–10 bullet points:"}
    ]

def prompt_quiz(ctx, n):
    return [
        {"role": "system", "content": "Create MCQs with answers."},
        {"role": "user", "content": f"Content:\n{ctx}\n\nWrite {n} MCQs:\nQ1: ...\nA)\nB)\nC)\nD)\nCorrect Answer:"}
    ]

# NEW: fallback doc classifier
def prompt_doc_classifier(full_text):
    return [
        {"role": "system", "content": "Identify the type of document (CV, resume, report, article, textbook, letter, etc). Be direct."},
        {"role": "user", "content": f"Here is the document:\n\n{full_text}\n\nWhat type of document is this?"}
    ]


# ---------- File Loader ----------
def load_file(upload):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.name)[1]) as tmp:
        tmp.write(upload.getvalue())
        path = tmp.name

    try:
        if upload.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        return loader.load()
    finally:
        os.unlink(path)


# ---------- Embeddings ----------
def get_embedder():
    if "embedder" not in st.session_state:
        st.session_state.embedder = SentenceTransformer(EMBED_MODEL)
    return st.session_state.embedder


# ---------- Qdrant ----------
def init_qdrant():
    st.session_state.qdrant = QdrantClient(":memory:")

def index_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
    chunks = splitter.split_documents(documents)
    texts = [c.page_content for c in chunks]

    embedder = get_embedder()
    vectors = embedder.encode(texts)

    qdrant = st.session_state.qdrant
    qdrant.recreate_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )

    points = [
        PointStruct(id=i, vector=vectors[i], payload={"text": texts[i]})
        for i in range(len(texts))
    ]

    qdrant.upsert("docs", points)
    st.session_state.texts = texts


def retrieve(query, top_k=3):
    embedder = get_embedder()
    q_vec = embedder.encode([query])[0]

    qdrant = st.session_state.qdrant
    results = qdrant.search("docs", query_vector=q_vec, limit=top_k)

    ctx = "\n\n".join(hit.payload["text"] for hit in results)
    return ctx[:MAX_CONTEXT_CHARS]


# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="AmbedkarGPT — Fast RAG", layout="wide")
st.title("📚 AmbedkarGPT — Fast RAG (Groq + Qdrant In-Memory)")
st.write("Stable, fast, cloud-friendly version.")

with st.sidebar:
    st.header("📂 Upload Document")
    upload = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if upload:
        if st.button("Process Document"):
            with st.spinner("Loading..."):
                docs = load_file(upload)
            with st.spinner("Indexing..."):
                init_qdrant()
                index_documents(docs)
            st.success("Document indexed!")


if "qdrant" not in st.session_state:
    st.info("Upload a document to begin.")
    st.stop()


# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A", "📝 Notes", "📋 Short Notes", "❓ Quiz"])

# Q&A
with tab1:
    q = st.text_input("Ask a question:")

    if q:
        ctx = retrieve(q)
        placeholder = st.empty()

        # If vague question, classify document type instead
        vague = any(v in q.lower().strip() for v in [
            "what is this", "what is this pdf", "what is this document",
            "what is this about", "describe this", "tell me about this pdf"
        ])

        with st.spinner("Thinking..."):
            if vague:
                full_text = "\n\n".join(st.session_state.texts)[:MAX_CONTEXT_CHARS]
                groq_stream(prompt_doc_classifier(full_text), placeholder)
            else:
                groq_stream(prompt_qa(ctx, q), placeholder)

# Notes
with tab2:
    if st.button("Generate Notes"):
        ctx = "\n\n".join(st.session_state.texts)[:MAX_CONTEXT_CHARS]
        placeholder = st.empty()
        with st.spinner("Generating notes..."):
            notes = groq_stream(prompt_notes(ctx), placeholder)
        st.download_button("Download Notes", notes, "notes.txt")

# Short Notes
with tab3:
    if st.button("Generate Short Notes"):
        ctx = "\n\n".join(st.session_state.texts)[:MAX_CONTEXT_CHARS]
        placeholder = st.empty()
        with st.spinner("Generating..."):
            s = groq_stream(prompt_short(ctx), placeholder)
        st.download_button("Download Summary", s, "short_notes.txt")

# Quiz
with tab4:
    n = st.slider("Number of questions", 3, 10, 5)
    if st.button("Generate Quiz"):
        ctx = "\n\n".join(st.session_state.texts)[:MAX_CONTEXT_CHARS]
        placeholder = st.empty()
        with st.spinner("Generating quiz..."):
            quiz = groq_stream(prompt_quiz(ctx, n), placeholder)
        st.download_button("Download Quiz", quiz, "quiz.txt")
