# RAG Chatbot with Live PDF Upload and Streaming Responses

## Overview

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that enables users to upload PDF documents, process them in real-time, and ask questions to receive context-aware answers. Built with **Streamlit** for a dynamic web interface, **LangChain** for the RAG pipeline, **Groq** for language model integration, **FAISS** for efficient vector storage, and **HuggingFace** embeddings for semantic retrieval, the chatbot supports live PDF uploads, streaming word-by-word responses, and conversational features like small talk and chat history clearing.

The project consists of three main scripts:
- `create_memory.py`: Processes PDFs from a directory and creates a persistent FAISS vector database.
- `connect_memory.py`: Command-line script to query the pre-built FAISS database.
- `main.py`: Streamlit web app for live PDF uploads, real-time querying, and conversational interactions.

## Features

- **Live PDF Upload**: Upload PDFs through the Streamlit UI, processed in-memory for real-time question answering.
- **Streaming Responses**: Displays answers word-by-word, mimicking ChatGPT’s conversational style, with an option for pause-and-display-all mode.
- **Conversational Features**: Handles small talk (e.g., "what's up?", "hello") with friendly responses and supports chat history clearing via text command ("clear") or button.
- **Semantic Retrieval**: Uses FAISS and HuggingFace’s `sentence-transformers/all-MiniLM-L6-v2` for accurate, context-aware answers from document chunks.
- **Robust Error Handling**: Manages invalid PDFs, missing databases, and query errors with user-friendly feedback.

## Prerequisites

- Python 3.8+
- A Groq API key (obtain from [Groq](https://console.groq.com/))
- Installed dependencies (see Installation below)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   