import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Hybrid Search RAG",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Hybrid Search RAG")
st.markdown("Combines **Semantic Search (FAISS)** + **Keyword Search (BM25)** using **Reciprocal Rank Fusion** for smarter answers!")

with st.sidebar:
    st.header("⚙️ Settings")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get free API key at console.groq.com"
    )
    model = st.selectbox("🤖 Model", [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
    ])
    st.divider()
    st.subheader("⚖️ Search Weights (RRF)")
    semantic_weight = st.slider(
        "Semantic Weight", 0.0, 1.0, 0.5, 0.1,
        help="Controls how much FAISS semantic search influences results"
    )
    keyword_weight = round(1.0 - semantic_weight, 1)
    col1, col2 = st.columns(2)
    col1.metric("🧠 Semantic", semantic_weight)
    col2.metric("🔤 Keyword", keyword_weight)
    st.divider()
    st.subheader("📖 How It Works")
    st.markdown("""
**🧠 Dense (FAISS):**
Finds semantically similar chunks using vector embeddings

**🔤 Sparse (BM25):**
Finds exact keyword matches using TF-IDF scoring

**✅ Hybrid RRF:**
Combines both using Reciprocal Rank Fusion:
`score = w × 1/(k + rank)`
""")
    st.divider()
    st.caption("Built with LangChain + Groq + HuggingFace")


def hybrid_search_rrf(query, faiss_retriever, bm25_retriever, semantic_weight=0.5, k=4, rrf_k=60):
    keyword_weight = 1.0 - semantic_weight
    faiss_docs = faiss_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)
    scores = {}
    doc_map = {}
    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content
        rrf_score = semantic_weight * (1 / (rrf_k + rank + 1))
        scores[key] = scores.get(key, 0) + rrf_score
        doc_map[key] = doc
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        rrf_score = keyword_weight * (1 / (rrf_k + rank + 1))
        scores[key] = scores.get(key, 0) + rrf_score
        doc_map[key] = doc
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    top_docs = [doc_map[key] for key in sorted_keys[:k]]
    top_scores = {key: round(scores[key], 6) for key in sorted_keys[:k]}
    return top_docs, top_scores


def get_answer(docs, question, llm):
    context = "\n\n".join([
        f"[Chunk {i+1}]: {doc.page_content}"
        for i, doc in enumerate(docs)
    ])
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Answer the question using ONLY the context provided.
Be specific and cite exact information from the context.
If the answer is not in the context, say 'Not found in document.'
Keep your answer clear and concise."""),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


st.subheader("📄 Step 1: Upload Document")
uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["txt", "pdf"],
    help="The document will be chunked and indexed for hybrid search"
)

if uploaded_file:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    else:
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            st.error(f"PDF error: {str(e)}")
            text = ""

    if text:
        word_count = len(text.split())
        st.success(f"✅ Document loaded: **{word_count} words**")
        with st.expander("👀 Preview Document"):
            st.text(text[:1000] + "..." if len(text) > 1000 else text)

        st.subheader("⚙️ Step 2: Build Search Indexes")
        if st.button("🔨 Build FAISS + BM25 Indexes", type="primary"):
            with st.spinner("⏳ Chunking document..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_text(text)
                docs = [Document(page_content=chunk) for chunk in chunks]
            st.info(f"📄 Split into **{len(docs)} chunks**")

            with st.spinner("🧠 Building FAISS semantic index..."):
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                faiss_vectorstore = FAISS.from_documents(docs, embeddings)
                faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})

            with st.spinner("🔤 Building BM25 keyword index..."):
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 4

            st.session_state.faiss_retriever = faiss_retriever
            st.session_state.bm25_retriever = bm25_retriever
            st.session_state.num_chunks = len(docs)
            st.success("✅ Both indexes built successfully!")

            col1, col2, col3 = st.columns(3)
            col1.metric("📚 Total Chunks", len(docs))
            col2.metric("🧠 FAISS Vectors", len(docs))
            col3.metric("🔤 BM25 Terms", len(docs))


if "faiss_retriever" in st.session_state:
    st.divider()
    st.subheader("❓ Step 3: Ask a Question")
    question = st.text_input("Your question:", placeholder="Ask anything about the document...")
    show_comparison = st.checkbox("📊 Show full comparison (Semantic vs Keyword vs Hybrid)", value=True)

    if st.button("🔍 Search & Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question!")
        elif not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar!")
        else:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name=model, temperature=0)

            if show_comparison:
                st.markdown("### 📊 Comparison: Three Search Methods")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 🧠 Semantic Only\n*(FAISS Vector Search)*")
                    with st.spinner("Searching semantically..."):
                        sem_docs = st.session_state.faiss_retriever.invoke(question)
                        sem_answer = get_answer(sem_docs, question, llm)
                    st.markdown("**Answer:**")
                    st.info(sem_answer)
                    with st.expander(f"📄 Sources ({len(sem_docs)} chunks)"):
                        for i, doc in enumerate(sem_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.caption(doc.page_content[:300] + "...")
                            st.divider()

                with col2:
                    st.markdown("#### 🔤 Keyword Only\n*(BM25 Sparse Search)*")
                    with st.spinner("Searching by keywords..."):
                        kw_docs = st.session_state.bm25_retriever.invoke(question)
                        kw_answer = get_answer(kw_docs, question, llm)
                    st.markdown("**Answer:**")
                    st.info(kw_answer)
                    with st.expander(f"📄 Sources ({len(kw_docs)} chunks)"):
                        for i, doc in enumerate(kw_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.caption(doc.page_content[:300] + "...")
                            st.divider()

                with col3:
                    st.markdown("#### ✅ Hybrid RRF\n*(Best of Both)*")
                    with st.spinner("Running hybrid RRF search..."):
                        hyb_docs, hyb_scores = hybrid_search_rrf(
                            question,
                            st.session_state.faiss_retriever,
                            st.session_state.bm25_retriever,
                            semantic_weight=semantic_weight
                        )
                        hyb_answer = get_answer(hyb_docs, question, llm)
                    st.markdown("**Answer:**")
                    st.success(hyb_answer)
                    with st.expander(f"📄 Sources + RRF Scores ({len(hyb_docs)} chunks)"):
                        for i, (doc, score) in enumerate(zip(hyb_docs, hyb_scores.values())):
                            st.markdown(f"**Chunk {i+1}** | RRF Score: `{score}`")
                            st.caption(doc.page_content[:300] + "...")
                            st.divider()

            else:
                with st.spinner("🔍 Running Hybrid RRF search..."):
                    hyb_docs, hyb_scores = hybrid_search_rrf(
                        question,
                        st.session_state.faiss_retriever,
                        st.session_state.bm25_retriever,
                        semantic_weight=semantic_weight
                    )
                    answer = get_answer(hyb_docs, question, llm)
                st.markdown("### ✅ Answer")
                st.success(answer)
                st.markdown("### 📄 Sources")
                for i, (doc, score) in enumerate(zip(hyb_docs, hyb_scores.values())):
                    with st.expander(f"Chunk {i+1} | RRF Score: {score}"):
                        st.write(doc.page_content)