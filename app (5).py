import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import streamlit as st

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions from Zyro Dynamics' "
    "internal policy documents. Your question appears to be outside my scope. "
    "For other matters, please contact the appropriate team."
)

RAG_SYSTEM = """You are an HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer employee questions ONLY using the context below.
Be concise and cite the policy document name when possible.
If the context does not contain enough information, say so clearly.

Context:
{context}"""

OOS_SYSTEM = """Classify the user question as HR-related or not.
HR topics: leave, salary, payroll, benefits, performance, WFH, attendance, onboarding,
resignation, code of conduct, POSH, travel reimbursements, IT policy, compensation, insurance.
Reply ONLY with IN_SCOPE or OUT_OF_SCOPE."""


def load_pdfs_manually():
    """Load all PDFs from current directory or any subfolder using pypdf directly."""
    import pypdf

    docs = []
    search_dirs = ["."]
    # also check one level of subfolders
    for entry in os.listdir("."):
        if os.path.isdir(entry) and not entry.startswith("."):
            search_dirs.append(entry)

    for search_dir in search_dirs:
        for fname in sorted(os.listdir(search_dir)):
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(search_dir, fname)
                try:
                    reader = pypdf.PdfReader(fpath)
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            docs.append({
                                "text": text,
                                "source": fname,
                                "page": i
                            })
                except Exception as e:
                    st.warning(f"Could not read {fname}: {e}")

    return docs


def chunk_docs(docs, chunk_size=800, overlap=150):
    """Split documents into overlapping chunks."""
    chunks = []
    for doc in docs:
        text = doc["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "page": doc["page"]
                })
            start += chunk_size - overlap
    return chunks


@st.cache_resource(show_spinner="Loading HR policy documents — please wait...")
def build_pipeline():
    # ── API key ────────────────────────────────────────────────────────────────
    groq_api_key = ""
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = os.getenv("GROQ_API_KEY", "")

    if not groq_api_key:
        st.error("Missing GROQ_API_KEY. Go to Streamlit Cloud → Settings → Secrets and add:\nGROQ_API_KEY = 'your-key-here'")
        st.stop()

    # ── Load PDFs ──────────────────────────────────────────────────────────────
    raw_docs = load_pdfs_manually()
    if not raw_docs:
        st.error(f"No PDFs found. Directory contents: {os.listdir('.')}")
        st.stop()

    # ── Chunk ──────────────────────────────────────────────────────────────────
    chunks = chunk_docs(raw_docs)

    # ── Embed ──────────────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # ── FAISS index ────────────────────────────────────────────────────────────
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity (normalized)
    index.add(embeddings.astype("float32"))

    # ── LLM ────────────────────────────────────────────────────────────────────
    from groq import Groq
    groq_client = Groq(api_key=groq_api_key)

    return model, index, chunks, groq_client


def retrieve(query, model, index, chunks, k=5):
    """Retrieve top-k relevant chunks using FAISS."""
    import numpy as np
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def call_llm(groq_client, system_prompt, user_message):
    """Call Groq LLM directly."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.1,
        max_tokens=512
    )
    return response.choices[0].message.content


def ask_bot(question, model, index, chunks, groq_client):
    # Guardrail
    cls = call_llm(groq_client, OOS_SYSTEM, question).strip().upper()
    if "OUT_OF_SCOPE" in cls:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "is_oos": True}

    # Retrieve
    retrieved = retrieve(question, model, index, chunks)
    context = "\n\n".join(
        f"[{c['source']}, p.{c['page']}]\n{c['text']}"
        for c in retrieved
    )

    # Generate
    answer = call_llm(groq_client, RAG_SYSTEM.format(context=context), question)
    sources = sorted({f"{c['source']} (p.{c['page']})" for c in retrieved})
    return {"answer": answer, "sources": sources, "is_oos": False}


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask any question about Zyro Dynamics HR policies.")
st.divider()

model, index, chunks, groq_client = build_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

if prompt := st.chat_input("Ask an HR question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Looking up HR policies..."):
            result = ask_bot(prompt, model, index, chunks, groq_client)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📄 Sources"):
                for s in result["sources"]:
                    st.markdown(f"- `{s}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })

with st.sidebar:
    st.header("About")
    st.markdown(
        "HR Help Desk for **Zyro Dynamics Pvt. Ltd.** "
        "powered by RAG over 11 internal policy documents."
    )
    st.divider()
    st.markdown("**Policies covered:**")
    for p in [
        "Company Profile", "Employee Handbook", "Leave Policy",
        "Work From Home Policy", "Code of Conduct", "Performance Review",
        "Compensation & Benefits", "IT & Data Security",
        "POSH Policy", "Onboarding & Separation", "Travel & Expense"
    ]:
        st.markdown(f"- {p}")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
