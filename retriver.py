# File: retriever.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict
from config import EMBEDDING_MODEL_NAME

class Retriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_stores = {}  # Dict of paper_id: FAISS vectorstore

    def add_paper(self, paper_id: str, vectorstore: FAISS):
        """Add processed paper to retriever."""
        self.vector_stores[paper_id] = vectorstore

    def semantic_search(self, query: str, paper_ids: List[str], k: int = 5) -> List[Dict]:
        """Semantic search across papers using LangChain retrievers."""
        all_docs = []
        for pid in paper_ids:
            if pid not in self.vector_stores:
                continue
            vectorstore = self.vector_stores[pid]
            docs = vectorstore.similarity_search_with_score(query, k=k)
            for doc, score in docs:
                all_docs.append({
                    'paper_id': pid,
                    'chunk': doc.page_content,
                    'score': score,  # Distance, lower is better
                    'metadata': doc.metadata
                })
        # Sort by score (ascending distance)
        all_docs.sort(key=lambda x: x['score'])
        return all_docs[:k]