# 🔍 Hybrid Search RAG

> A production-deployed RAG application combining Semantic Search (FAISS) + Keyword Search (BM25) using Reciprocal Rank Fusion for smarter document Q&A.

🚀 **Live Demo:** [hybrid-search-rag-vijaya.streamlit.app](https://hybrid-search-rag-vijaya.streamlit.app)

---

## 📌 Overview

Hybrid Search RAG is a Retrieval-Augmented Generation (RAG) application that lets users upload PDF or TXT documents and ask questions. Unlike traditional RAG systems that rely on a single retrieval method, this app combines **dense semantic search** (FAISS) and **sparse keyword search** (BM25) using **Reciprocal Rank Fusion (RRF)** — giving you the best of both worlds.

It also displays a **side-by-side comparison** of all three retrieval methods so you can visually compare how Semantic Search, Keyword Search, and Hybrid RRF perform for the same query.

---

## ✨ Features

- 📄 **PDF & TXT Upload** — Upload any document and extract text automatically
- 🧠 **Semantic Search (FAISS)** — Finds contextually similar chunks using HuggingFace vector embeddings
- 🔤 **Keyword Search (BM25)** — Retrieves relevant chunks based on keyword frequency and term importance
- ✅ **Hybrid RRF Search** — Combines both methods using Reciprocal Rank Fusion formula
- 📊 **3-Column Comparison** — Side-by-side view of Semantic vs Keyword vs Hybrid results
- ⚖️ **Adjustable Weights** — Slider to control semantic vs keyword influence in RRF scoring
- 🔢 **RRF Scores Displayed** — See the actual mathematical scores for each retrieved chunk
- 🤖 **Multiple LLM Models** — Switch between GPT-OSS 20B and 120B via Groq API

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Web UI framework |
| **LangChain** | LLM orchestration & retrieval |
| **FAISS** | Dense vector search index |
| **BM25** | Sparse keyword search |
| **HuggingFace Embeddings** | Text vectorization (all-MiniLM-L6-v2) |
| **Groq API** | Fast LLM inference (free tier available) |
| **PyPDF2** | PDF text extraction |
| **Streamlit Cloud** | Production deployment |

---

## 🏗️ Architecture

    User Uploads Document (PDF / TXT)
                  ↓
           Text Extraction
                  ↓
        RecursiveCharacterTextSplitter
        (chunk_size=500, overlap=50)
                  ↓
             ┌────┴────┐
             ↓         ↓
       FAISS Index   BM25 Index
      (Dense/Semantic) (Sparse/Keyword)
             ↓         ↓
        ┌────┴─────────┴────┐
        │  Reciprocal Rank  │
        │  Fusion (RRF)     │
        │  score = w × 1/(k+rank) │
        └────────┬──────────┘
                 ↓
         Top-K Hybrid Chunks
                 ↓
            Groq LLM
        (openai/gpt-oss-20b)
                 ↓
           Final Answer

---

## 🧮 How Reciprocal Rank Fusion Works

RRF combines rankings from multiple retrieval systems using the formula:

    RRF_score = semantic_weight × 1/(rrf_k + rank_faiss)
               + keyword_weight  × 1/(rrf_k + rank_bm25)

Where:
- `semantic_weight + keyword_weight = 1.0` (controlled by the slider)
- `rrf_k = 60` (smoothing constant to reduce the impact of top ranks)
- Documents appearing in **both** result sets get a **combined score boost**

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Groq API Key (free at [console.groq.com](https://console.groq.com))

### Installation

    git clone https://github.com/vijaya-842/Hybrid-Search-Rag.git
    cd Hybrid-Search-Rag
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

### Configuration

Create a `.env` file:

    GROQ_API_KEY=your_groq_api_key_here

### Run Locally

    streamlit run app.py

---

## 📦 Requirements

    langchain-core
    langchain-community
    langchain-groq
    langchain-huggingface
    langchain-text-splitters
    streamlit
    python-dotenv
    faiss-cpu
    sentence-transformers
    rank-bm25
    PyPDF2

---

## 💡 How to Use

1. Enter your Groq API key in the sidebar
2. Upload a PDF or TXT file
3. Click **"Build FAISS + BM25 Indexes"**
4. Type your question
5. Click **"Search & Answer"** to see all 3 methods side by side
6. Adjust the semantic weight slider to change RRF scores

---

## 🌐 Deployment

Deployed on **Streamlit Cloud**. To deploy your own:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GROQ_API_KEY` in Secrets
5. Deploy!

---

## 🧠 What I Learned

- How **FAISS** builds and queries dense vector indexes
- How **BM25** retrieves documents based on keyword frequency and term importance
- The mathematics behind **Reciprocal Rank Fusion (RRF)**
- How combining retrievers affects result quality depending on the query
- How to build **multi-retriever RAG pipelines** with LangChain
- Deploying a **production RAG app** on Streamlit Cloud

---

## 👩‍💻 Author

**Vijaya Lakshmi Atluri**
- GitHub: [@vijaya-842](https://github.com/vijaya-842)
- LinkedIn: https://www.linkedin.com/in/vijaya-atluri/

---

## 📄 License

MIT License

---

⭐ If you found this helpful, please star the repo!
