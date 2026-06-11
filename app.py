import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

CORPUS_PATH = "hr_docs/"

# ── Debug: show what files Streamlit can see ───────────────────────────────────
with st.expander("🔍 Debug — click to see file structure (remove after fixing)", expanded=False):
    import glob
    st.code(f"Current working directory: {os.getcwd()}")
    st.code("All files in working directory:\n" + "\n".join(sorted(glob.glob("**/*", recursive=True)[:50])))
    st.code(f"hr_docs/ exists: {os.path.exists('hr_docs/')}")
    st.code(f"hr_docs/ contents: {os.listdir('hr_docs/') if os.path.exists('hr_docs/') else 'FOLDER NOT FOUND'}")

REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions from Zyro Dynamics' "
    "internal policy documents. Your question appears to be outside my scope. "
    "For other matters, please contact the appropriate team."
)

@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_pipeline():
    if not os.path.exists(CORPUS_PATH):
        st.error(
            "Missing 'hr_docs/' folder. "
            "Please create an 'hr_docs/' folder in your repo and add all 11 HR policy PDFs inside it."
        )
        st.stop()

    # Check both Streamlit secrets and environment variables
    groq_api_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        st.error("Missing GROQ_API_KEY. Go to Streamlit Cloud → App Settings → Secrets and add it.")
        st.stop()

    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    if not documents:
        st.error("No PDF documents found inside 'hr_docs/'. Please add your HR policy PDFs.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Use CPU-only, no torchvision needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512
    )

    return retriever, llm


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'Unknown')}, Page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer employee questions ONLY using the context below.
Be concise and cite the policy document name and page number when possible.
If the context does not contain sufficient information, say so clearly.

Context:
{context}"""
    ),
    ("human", "{question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Classify the user question as HR-related or not.
HR topics: leave, salary, payroll, benefits, performance, WFH, attendance, onboarding,
resignation, code of conduct, POSH, travel reimbursements, IT policy, compensation, insurance.
Reply ONLY with IN_SCOPE or OUT_OF_SCOPE."""
    ),
    ("human", "{question}")
])


def ask_bot(question, retriever, llm):
    cls_prompt = OOS_PROMPT.invoke({"question": question})
    cls = StrOutputParser().invoke(llm.invoke(cls_prompt)).strip().upper()

    if "OUT_OF_SCOPE" in cls:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "is_oos": True}

    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt_val = RAG_PROMPT.invoke({"context": context, "question": question})
    answer = StrOutputParser().invoke(llm.invoke(prompt_val))

    sources = sorted({
        f"{os.path.basename(d.metadata.get('source', 'Unknown'))} (Page {d.metadata.get('page', '?')})"
        for d in docs
    })

    return {"answer": answer, "sources": sources, "is_oos": False}


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask any question about Zyro Dynamics HR policies — leave, salary, performance, WFH, and more.")
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
        "This chatbot answers HR policy questions for **Zyro Dynamics Pvt. Ltd.** "
        "using Retrieval-Augmented Generation (RAG) over internal policy documents."
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
