# Zyro Dynamics HR Help Desk — RAG Chatbot

An AI-powered HR Help Desk chatbot built with RAG (Retrieval-Augmented Generation)
over 11 internal HR policy documents.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  Guardrail (LLM)    │  ← classifies IN_SCOPE / OUT_OF_SCOPE
└─────────────────────┘
     │ IN_SCOPE
     ▼
┌─────────────────────┐
│  MMR Retriever      │  ← FAISS vector store, top-5 diverse chunks
│  (FAISS + MMR)      │
└─────────────────────┘
     │ retrieved docs
     ▼
┌─────────────────────┐
│  RAG Chain (LLM)    │  ← answers strictly from retrieved context
│  Groq / Llama 3.1   │
└─────────────────────┘
     │
     ▼
Answer + Source Citations
```

---

## Files

| File | Description |
|------|-------------|
| `Completed_Notebook.ipynb` | Fully filled-in Kaggle notebook (all TODO cells solved) |
| `app.py` | Streamlit chatbot application |
| `requirements.txt` | Python dependencies for Streamlit Cloud |

---

## Running the Notebook (Kaggle)

1. Upload `Completed_Notebook.ipynb` to Kaggle
2. Add secrets: `GROQ_API_KEY`, `LANGCHAIN_API_KEY`
3. Run all cells top-to-bottom
4. After Cell 12 test questions run, get your LangSmith trace URL
5. Run Cell 16 to generate `submission.csv`

---

## Deploying the Streamlit App

### Step 1 — Prepare files
Create a GitHub repo with:
```
├── app.py
├── requirements.txt
└── hr_docs/
    ├── 00_Company_Profile.pdf
    ├── 01_Employee_Handbook.pdf
    ├── 02_Leave_Policy.pdf
    ├── 03_Work_From_Home_Policy.pdf
    ├── 04_Code_of_Conduct.pdf
    ├── 05_Performance_Review_Policy.pdf
    ├── 06_Compensation_and_Benefits_Policy.pdf
    ├── 07_IT_and_Data_Security_Policy.pdf
    ├── 08_Prevention_of_Sexual_Harassment_Policy.pdf
    ├── 09_Onboarding_and_Separation_Policy.pdf
    └── 10_Travel_and_Expense_Policy.pdf
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Connect your GitHub repo
3. Set `app.py` as the main file
4. Add secret: `GROQ_API_KEY = "your-key-here"` under **Settings → Secrets**
5. Deploy — your URL will be `https://your-app-name.streamlit.app`

---

## Key Design Decisions

### Chunking
- **chunk_size = 800** — large enough to capture complete policy clauses
- **chunk_overlap = 150** — preserves context across chunk boundaries

### Retrieval
- **MMR (Maximal Marginal Relevance)** — balances relevance and diversity
- **k=5, fetch_k=20** — considers 20 candidates, returns 5 diverse results
- **lambda_mult=0.7** — slightly favour relevance over diversity

### Guardrails
- A separate LLM call classifies every question before RAG
- OUT_OF_SCOPE → polite refusal, no RAG invoked
- Prevents hallucination on unrelated topics

### LLM
- **Groq + Llama 3.1 8B Instant** — free tier, very fast inference
- **temperature=0.1** — near-deterministic answers, minimal creativity/hallucination

---

## Scoring Notes

- **Q01-Q10** (in-scope): grounded answers from retrieved policy chunks
- **Q11-Q15** (out-of-scope): guardrail fires, returns polite refusal message
