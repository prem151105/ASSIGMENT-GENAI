"""
app.py
------
Streamlit UI for the RAG Chatbot.

Steps covered (5 & 6 of the assignment):
  5. Retrieve + Generate -> Wire the FAISS retriever into a
     RetrievalQA chain powered by Groq (free, fast LLM API).
  6. Test -> Ask questions in the chat box and verify answers
     are grounded in the uploaded document. If the answer isn't
     in the document, the bot replies "Not found in document".

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"

# ----------------------------------------------------------------------
# Prompt: forces the model to answer ONLY from the provided context and
# to say "Not found in document" if the answer isn't there.
# ----------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions
using ONLY the context below, which was retrieved from the user's own
document.

Rules:
- Answer ONLY using information present in the context.
- If the answer is not contained in the context, reply exactly:
  "Not found in document"
- Do not use any outside knowledge. Do not make anything up.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore_from_file(uploaded_file, embeddings):
    """Save the uploaded file to a temp path, load + split + embed it."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding="utf-8")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


def load_default_vectorstore(embeddings):
    """Load a previously-saved FAISS index from disk, if it exists."""
    if os.path.exists(INDEX_DIR):
        return FAISS.load_local(
            INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
    return None


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot — Ask Questions About Your Document")
st.caption(
    "Built with LangChain + FAISS + HuggingFace embeddings + Groq LLM. "
    "Answers come only from the uploaded document."
)

embeddings = get_embeddings()

# Sidebar: upload / index controls
with st.sidebar:
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        with st.spinner("Reading, chunking and embedding your document..."):
            vectorstore, n_chunks = build_vectorstore_from_file(uploaded_file, embeddings)
            st.session_state.vectorstore = vectorstore
        st.success(f"Indexed {uploaded_file.name} into {n_chunks} chunks ✅")

    elif "vectorstore" not in st.session_state:
        with st.spinner("Loading default index..."):
            vs = load_default_vectorstore(embeddings)
            if vs is not None:
                st.session_state.vectorstore = vs
                st.info("Loaded existing FAISS index from `faiss_index/`.")
            else:
                st.warning(
                    "No index found. Upload a PDF/TXT above, or run "
                    "`python ingest.py` first (it will index files in `data/`)."
                )

    st.divider()
    st.markdown(
        "**Free setup:**\n"
        "- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, free)\n"
        "- LLM: Groq (`llama-3.1-8b-instant`, free tier)\n\n"
        "Set `GROQ_API_KEY` in a `.env` file. See `.env.example`."
    )

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Chat input
question = st.chat_input("Ask a question about your document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if "vectorstore" not in st.session_state:
            answer = "Please upload a document first (see sidebar)."
            sources = []
        else:
            llm = get_llm()
            if llm is None:
                answer = (
                    "⚠️ GROQ_API_KEY not set. Add it to a `.env` file "
                    "(see `.env.example`) to enable answers."
                )
                sources = []
            else:
                with st.spinner("Retrieving relevant chunks and generating answer..."):
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    )
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": QA_PROMPT},
                    )
                    result = qa_chain.invoke({"query": question})
                    answer = result["result"]
                    sources = [
                        f"Page {d.metadata.get('page', '-')} — "
                        f"{d.metadata.get('source', 'document')}"
                        for d in result["source_documents"]
                    ]

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
