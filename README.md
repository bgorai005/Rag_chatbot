# AI-Powered Research Summarizer

## Overview

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that enables users to upload PDF documents, process them in real-time, and ask questions to receive context-aware answers. Built with **Streamlit** for a dynamic web interface, **LangChain** for the RAG pipeline, **Groq** for language model integration, **FAISS** for efficient vector storage, and **HuggingFace** embeddings for semantic retrieval, the chatbot supports live PDF uploads, streaming word-by-word responses, and conversational features like small talk and chat history clearing.

The project consists of three main scripts:
- `create_memory.py`: Processes PDFs from a directory and creates a persistent FAISS vector database.
- `connect_memory.py`: Command-line script to query the pre-built FAISS database.
- `main.py`: Streamlit web app for live PDF uploads, real-time querying, and conversational interactions.

## Features
- **Ingestion**: Search arXiv API or upload PDFs; persistent storage in `./documents`.
- **Processing**: Chunking, embedding (HuggingFace MiniLM), and FAISS indexing per paper.
- **RAG Pipeline**: Semantic search (k=3-5 chunks) + Llama 3 70B generation for accurate, cited responses.
- **Insights**: Grounded Q&A, section-wise summaries, key highlights, and multi-paper comparisons.
- **Export**: Structured Markdown/PDF reports for sharing.
- **Viz**: Sidebar charts for retrieval relevance scores.

## Tech Stack
- **Frontend**: Streamlit (interactive tabs, session state).
- **RAG Core**: LangChain (chains, splitters, embeddings), FAISS (vector store).
- **LLM**: Groq (Llama 3 70B, low-latency inference).
- **Ingestion**: arXiv API, PyMuPDF (PDF extraction).
- **Export**: FPDF (simple PDF gen).
- **Config**: dotenv for API keys; modular Python files.

## Quick Start

### Prerequisites
- Python 3.10+
- Groq API key (free tier: [console.groq.com](https://console.groq.com))

### Installation
1. Clone the repo:
## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   
