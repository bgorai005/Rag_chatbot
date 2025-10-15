# File: ingestion.py
import arxiv
import requests
import os
from typing import List, Dict
from langchain_community.document_loaders import PyMuPDFLoader
from config import DOCS_DIR

def fetch_arxiv_papers(query: str, max_results: int = 5) -> List[Dict]:
    """Fetch papers from arXiv API."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    papers = []
    for result in search.results():
        paper_info = {
            'title': result.title,
            'authors': [a.name for a in result.authors],
            'summary': result.summary,
            'pdf_url': result.pdf_url,
            'published': result.published,
            'id': result.entry_id.split('/')[-1]
        }
        papers.append(paper_info)
    return papers

def download_pdf(pdf_url: str, paper_id: str) -> str:
    """Download PDF to persistent docs directory."""
    pdf_path = os.path.join(DOCS_DIR, f"{paper_id}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path  # Already downloaded
    response = requests.get(pdf_url)
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    return pdf_path

def load_pdf_as_documents(pdf_path: str) -> List:
    """Load PDF using LangChain PyMuPDFLoader."""
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()