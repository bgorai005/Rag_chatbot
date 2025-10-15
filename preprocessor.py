# File: preprocessor.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Dict
from config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP

class Preprocessor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def split_documents(self, documents: List) -> List:
        """Split documents into chunks using LangChain splitter."""
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, chunks: List, metadata: Dict) -> FAISS:
        """Create FAISS vector store using LangChain."""
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'paper_id': metadata['id'],
                'section': 'full'  # Can be enhanced
            })
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return vectorstore