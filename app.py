import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# ── Find hr_docs folder anywhere in the repo ──────────────────────────────────
def find_corpus():
    """Search for PDFs - checks root dir first, then any subfolder."""
    # Check root directory directly
    root_pdfs = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    if root_pdfs:
        return ".", root_pdfs

    # Check one level of subfolders
    for entry in os.listdir("."):
        if os.path.isdir(entry) and not entry.startswith("."):
            pdfs = [f for f in os.listdir(entry) if f.lower().endswith(".pdf")]
            if pdfs:
                return entry, pdfs

    return None, []


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


@st.cache_resource(show_spinner="Loading HR policy documents — please wait...")
def build_pipeline():
    # ── 1. Locate PDFs ─────────────────────────────────────────────────────────
    corpus_path, pdf_files = find_corpus()

    if corpus_path is None:
        st.error(
            "Cannot find HR policy PDFs. "
            "Make sure your GitHub repo has an 'hr_docs/' folder containing the PDF files. "
            f"Current directory contents: {os.listdir('.')}"
        )
        st.stop()

    # ── 2. API key ─────────────────────────────────────────────────────────────
    groq_api_key = ""
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = os.getenv("GROQ_API_KEY", "")

    if not groq_api_key:
        st.error(
            "GROQ_API_KEY not found. "
            "In Streamlit Cloud go to: App menu → Settings → Secrets and add:\n"
            "GROQ_API_KEY = 'your-key-here'"
        )
        st.stop()

    # ── 3. Load PDFs ───────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    loader = PyPDFDirectoryLoader(corpus_path)
    documents = loader.load()

    # ── 4. Chunk ───────────────────────────────────────────────────────────────
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # ── 5. Embed (CPU only, no torchvision) ────────────────────────────────────
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # ── 6. FAISS vector store ──────────────────────────────────────────────────
    from langchain_community.vectorstores import FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # ── 7. LLM ─────────────────────────────────────────────────────────────────
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512
    )

    return retriever, llm


def format_docs(docs):
    return "\n\n".join(
        f"[{os.path.basename(d.metadata.get('source','?'))}, p.{d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )


def ask_bot(question, retriever, llm):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    parser = StrOutputParser()

    # Guardrail
    oos_prompt = ChatPromptTemplate.from_messages([
        ("system", OOS_SYSTEM),
        ("human", "{question}")
    ])
    cls = parser.invoke(llm.invoke(oos_prompt.invoke({"question": question}))).strip().upper()

    if "OUT_OF_SCOPE" in cls:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "is_oos": True}

    # RAG
    docs = retriever.invoke(question)
    context = format_docs(docs)

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM),
        ("human", "{question}")
    ])
    answer = parser.invoke(llm.invoke(rag_prompt.invoke({"context": context, "question": question})))
    sources = sorted({
        f"{os.path.basename(d.metadata.get('source','?'))} (p.{d.metadata.get('page','?')})"
        for d in docs
    })
    return {"answer": answer, "sources": sources, "is_oos": False}


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask any question about Zyro Dynamics HR policies.")
st.divider()

retriever, llm = build_pipeline()

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
            result = ask_bot(prompt, retriever, llm)
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
