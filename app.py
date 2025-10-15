# File: app.py
import streamlit as st
from ingenstion import fetch_arxiv_papers, download_pdf, load_pdf_as_documents
from preprocessor import Preprocessor
from retriver import Retriever
from generator import Generator
from exporter import export_to_pdf
from config import DOCS_DIR
import os
from typing import List, Dict
from langchain_core.documents import Document

st.set_page_config(page_title="AI Research Summarizer", layout="wide")

@st.cache_resource
def load_models():
    preprocessor = Preprocessor()
    retriever = Retriever()
    generator = Generator()
    return preprocessor, retriever, generator

preprocessor, retriever, generator = load_models()

# Session state for processed papers
if 'processed_papers' not in st.session_state:
    st.session_state.processed_papers = []
if 'paper_info' not in st.session_state:
    st.session_state.paper_info = {}

st.title("AI-Powered Research Summarizer")

# Info about .env and docs dir
with st.sidebar:
    st.info(f" Documents stored in: {DOCS_DIR}")

tab1, tab2, tab3, tab4 = st.tabs(["Ingestion", "Processing", "Q&A & Summary", "Compare & Export"])

with tab1:
    st.header("Fetch Papers from arXiv")
    query = st.text_input("Enter research query:")
    max_results = st.slider("Max results", 1, 10, 5)
    if st.button("Fetch Papers"):
        papers = fetch_arxiv_papers(query, max_results)
        st.session_state.fetched_papers = papers
        for paper in papers:
            st.write(f"**{paper['title']}** by {', '.join(paper['authors'])}")
            st.write(paper['summary'][:200] + "...")

    st.header("Manual Upload")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        # Save upload to persistent dir
        upload_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.uploaded_paper = {
            'title': uploaded_file.name,
            'id': uploaded_file.name.replace('.pdf', ''),
            'upload_path': upload_path
        }
        st.success(f"Uploaded to {upload_path}")

with tab2:
    st.header("Process Papers")
    if 'fetched_papers' in st.session_state:
        for paper in st.session_state.fetched_papers:
            if st.button(f"Process {paper['title'][:50]}..."):
                with st.spinner("Downloading and processing..."):
                    pdf_path = download_pdf(paper['pdf_url'], paper['id'])
                    documents = load_pdf_as_documents(pdf_path)
                    chunks = preprocessor.split_documents(documents)
                    vectorstore = preprocessor.create_vector_store(chunks, paper)
                    retriever.add_paper(paper['id'], vectorstore)
                    st.session_state.processed_papers.append(paper['id'])
                    st.session_state.paper_info[paper['id']] = paper
                    st.success(f"Processed {paper['title']} (PDF: {pdf_path})")

    if 'uploaded_paper' in st.session_state:
        paper = st.session_state.uploaded_paper
        if st.button(f"Process {paper['title']}"):
            with st.spinner("Processing..."):
                documents = load_pdf_as_documents(paper['upload_path'])
                chunks = preprocessor.split_documents(documents)
                vectorstore = preprocessor.create_vector_store(chunks, paper)
                retriever.add_paper(paper['id'], vectorstore)
                st.session_state.processed_papers.append(paper['id'])
                st.session_state.paper_info[paper['id']] = paper
                st.success(f"Processed {paper['title']} (PDF: {paper['upload_path']})")

with tab3:
    st.header("Q&A and Summarization")
    selected_papers = st.multiselect("Select papers for Q&A", st.session_state.processed_papers)
    question = st.text_input("Ask a question:")
    if st.button("Get Answer") and selected_papers:
        # For multi-paper, use combined search
        context_results = retriever.semantic_search(question, selected_papers, k=3)
        answer = generator.qa_pipeline(question, context_results)
        st.write("**Answer:**")
        st.write(answer)

    st.subheader("Summarize Paper")
    paper_to_summ = st.selectbox("Select paper", st.session_state.processed_papers)
    section_wise = st.checkbox("Section-wise summary")
    if st.button("Summarize"):
        if paper_to_summ in retriever.vector_stores:
            vectorstore = retriever.vector_stores[paper_to_summ]
            # Retrieve full text approx
            full_docs = vectorstore.similarity_search(" ", k=100)  # Large k to get most docs
            text = "\n".join([doc.page_content for doc in full_docs])
            summary = generator.summarize_paper(text, section_wise)
            st.write("**Summary:**")
            st.write(summary)

    st.subheader("Highlights")
    if paper_to_summ and st.button("Get Highlights"):
        if paper_to_summ in retriever.vector_stores:
            vectorstore = retriever.vector_stores[paper_to_summ]
            full_docs = vectorstore.similarity_search(" ", k=100)
            text = "\n".join([doc.page_content for doc in full_docs])
            highlights = generator.highlights(text)
            st.write("**Highlights:**")
            st.write(highlights)

with tab4:
    st.header("Compare Papers")
    papers_to_compare = st.multiselect("Select papers to compare", st.session_state.processed_papers, max_selections=3)
    if st.button("Compare") and len(papers_to_compare) >= 2:
        summaries = []
        for pid in papers_to_compare:
            if pid in retriever.vector_stores:
                vectorstore = retriever.vector_stores[pid]
                full_docs = vectorstore.similarity_search(" ", k=100)
                text = "\n".join([doc.page_content for doc in full_docs])
                summ = generator.summarize_paper(text, False)
                summaries.append(summ)
        comparison = generator.compare_papers(summaries)
        st.write("**Comparison:**")
        st.write(comparison)

    st.header("Export Report")
    report_content = st.text_area("Enter report Markdown (or use generated content)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export to PDF"):
            pdf_path = export_to_pdf(report_content)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f.read(), file_name="research_report.pdf")
    with col2:
        if st.button("Export to Markdown"):
            st.download_button("Download Markdown", report_content, file_name="research_report.md")

# Visualizations (simple example)
st.sidebar.header("Semantic Search Visualization")
if st.sidebar.button("Show Top Chunks"):
    if 'processed_papers' in st.session_state:
        sample_query = "example query"
        results = retriever.semantic_search(sample_query, st.session_state.processed_papers[:2], k=3)
        scores = [1 - res['score'] for res in results]  # Convert distance to similarity
        papers = [res['paper_id'] for res in results]
        st.sidebar.bar_chart({p: s for p, s in zip(papers, scores)})