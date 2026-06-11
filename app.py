import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# ── Constants ──────────────────────────────────────────────────────────────────
CORPUS_PATH = "hr_docs/"   # place your 11 PDFs in this folder before deploying
REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions from Zyro Dynamics' "
    "internal policy documents. Your question appears to be outside my scope. "
    "For other matters, please contact the appropriate team."
)

# ── Build RAG pipeline (cached so it runs only once per session) ───────────────
@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_pipeline():
    # 1. Load PDFs
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # 3. Embed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 4. Vector store with MMR retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    )

    # 5. LLM
    llm = ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
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
    ("system",
     """You are an HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer employee questions ONLY using the context below. Be concise and cite the policy document.
If the context does not contain sufficient information, say so clearly.

Context:
{context}"""),
    ("human", "{question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """Classify the user question as HR-related or not.
HR topics: leave, salary, payroll, benefits, performance, WFH, attendance, onboarding,
resignation, code of conduct, POSH, travel reimbursements, IT policy, compensation, insurance.
Reply ONLY with IN_SCOPE or OUT_OF_SCOPE."""),
    ("human", "{question}")
])


def ask_bot(question, retriever, llm):
    # Guardrail
    cls_prompt = OOS_PROMPT.invoke({"question": question})
    cls = StrOutputParser().invoke(llm.invoke(cls_prompt)).strip().upper()
    if "OUT_OF_SCOPE" in cls:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "is_oos": True}

    # RAG
    docs = retriever.invoke(question)
    context = format_docs(docs)
    prompt_val = RAG_PROMPT.invoke({"context": context, "question": question})
    answer = StrOutputParser().invoke(llm.invoke(prompt_val))
    sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
    return {"answer": answer, "sources": sources, "is_oos": False}


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask any question about Zyro Dynamics HR policies — leave, salary, performance, WFH, and more.")
st.divider()

# Load pipeline
retriever, llm = build_pipeline()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

# Chat input
if prompt := st.chat_input("Ask an HR question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
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

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown(
        "This chatbot answers HR policy questions for **Zyro Dynamics Pvt. Ltd.** "
        "using Retrieval-Augmented Generation (RAG) over 11 internal policy documents."
    )
    st.divider()
    st.markdown("**Policies covered:**")
    policies = [
        "Company Profile", "Employee Handbook", "Leave Policy",
        "Work From Home Policy", "Code of Conduct", "Performance Review",
        "Compensation & Benefits", "IT & Data Security",
        "POSH Policy", "Onboarding & Separation", "Travel & Expense"
    ]
    for p in policies:
        st.markdown(f"- {p}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()