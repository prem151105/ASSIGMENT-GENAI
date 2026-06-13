# 📚 RAG Chatbot with LangChain + FAISS + Groq

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions
**only from your own document** (PDF or TXT). If the answer isn't in the
document, it replies `"Not found in document"` instead of hallucinating.

Built for the **Day 3 Take-Home Assignment**: tokenization → embeddings →
prompting → RAG → LangChain, in one working project.

---

## ✨ Stack (100% free option)

| Step | Component |
|------|-----------|
| Load | `PyPDFLoader` / `TextLoader` |
| Split | `RecursiveCharacterTextSplitter` |
| Embed | `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`, runs locally, free) |
| Vector Store | `FAISS` (local, free) |
| Retrieve + Generate | `RetrievalQA` + `ChatGroq` (free-tier LLM API) |
| UI | `Streamlit` |

---

## 🗂️ Project Structure

```
rag-chatbot/
├── app.py              # Streamlit chatbot UI
├── ingest.py           # Builds FAISS index from files in data/
├── data/
│   └── sample_notes.txt   # Sample notes (OS scheduling) for testing
├── faiss_index/         # Generated vector index (created after ingest.py)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd rag-chatbot
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key

1. Go to [console.groq.com/keys](https://console.groq.com/keys) and create a free API key.
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Paste your key into `.env`:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```

### 4. (Optional) Build the index from your own notes

Drop your own `.pdf` or `.txt` notes into the `data/` folder (a sample
`sample_notes.txt` is already included), then run:

```bash
python ingest.py
```

This creates `faiss_index/` on disk.

### 5. Run the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

You can either:
- Use the pre-built `faiss_index/` (from step 4), **or**
- Upload a PDF/TXT directly in the sidebar — it will be chunked and embedded
  on the fly.

---

## 🧪 Testing (Step 6 of the assignment)

Try asking 5 questions and verify the answers are grounded in the document:

1. **A question clearly answered in the document** → should give a correct,
   specific answer with sources shown in the "Sources" expander.
2. **A follow-up detail question** → answer should match the document text.
3. **A question about something NOT in the document** (e.g. "What is the
   capital of France?" if using `sample_notes.txt`) → should reply
   `"Not found in document"`.
4. **A summary-style question** (e.g. "Summarize this document in 3 points")
   → should produce a grounded summary.
5. **A trick question that sounds plausible but isn't covered** → should
   still say `"Not found in document"` instead of guessing.

Example questions for `sample_notes.txt`:
- "What is the convoy effect?"
- "What are the four conditions necessary for a deadlock?"
- "Explain Round Robin scheduling."
- "What is a medium-term scheduler?"
- "What is the time complexity of quicksort?" → should return
  `"Not found in document"`.

---

## 📤 Pushing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: RAG chatbot with LangChain, FAISS, Groq"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

> Note: `.env` is excluded via `.gitignore` — never commit your API key.

---

## 📝 Submission Checklist

- [ ] GitHub repo link with this code pushed
- [ ] Short demo video/screenshots showing:
  - [ ] Uploading or indexing a document
  - [ ] At least 5 Q&A exchanges
  - [ ] At least one "Not found in document" response
  - [ ] Sources panel showing retrieved chunks

---

## 🔧 Customization Ideas (Bonus)

- Swap `llama-3.1-8b-instant` for `llama-3.3-70b-versatile` in `app.py` for
  higher-quality answers (still free on Groq's free tier, just slower).
- Add a "confidence" indicator using FAISS similarity scores
  (`vectorstore.similarity_search_with_score`).
- Persist chat history to a file or database.
- Support multiple documents and let users pick which one to query.
