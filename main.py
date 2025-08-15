import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
import time
import tempfile

# Load environment variables
load_dotenv()

# Initialize embedding model
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to process uploaded PDF
def process_pdf(uploaded_file):
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Create chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(documents)

        # Clean up temporary file
        os.unlink(tmp_file_path)

        return text_chunks
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# Function to create or update FAISS index
@st.cache_resource
def get_vectorstore(uploaded_file_key=None):
    embedding_model = get_embedding_model()
    
    # If no uploaded file, try to load existing FAISS database
    if uploaded_file_key is None:
        try:
            DB_FAISS_PATH = "vectorstore/db_faiss"
            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            return db
        except Exception as e:
            st.warning("No existing FAISS database found. Please upload a PDF.")
            return None
    
    # Process uploaded PDF
    text_chunks = process_pdf(st.session_state.uploaded_file)
    if text_chunks:
        # Create new in-memory FAISS index
        db = FAISS.from_documents(text_chunks, embedding_model)
        return db
    return None

# Custom prompt template
def set_custom_prompt():
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know — don't make up an answer.
    Only use the given context.

    Context: {context}
    Question: {question}

    Answer:
    """
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Simple small-talk detection
def is_small_talk(user_input):
    small_talk_keywords = [
        "hi", "hello", "hey", "thanks", "thank you", "good morning", 
        "good evening", "how are you", "what's up", "how's it going", 
        "sup", "yo", "howdy"
    ]
    return any(user_input.lower().strip().startswith(kw) for kw in small_talk_keywords)

# Clear chat detection
def is_clear_chat(user_input):
    clear_chat_keywords = ["clear", "clear all", "clear chat", "reset", "reset chat"]
    return any(user_input.lower().strip() == kw for kw in clear_chat_keywords)

def stream_response(llm, retriever, query, prompt_template):
    """Stream response token by token."""
    context = ""
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    if docs:
        context = "\n".join([doc.page_content for doc in docs[:3]])
    else:
        context = "No relevant information found."

    # Format prompt
    formatted_prompt = prompt_template.format(context=context, question=query)
    
    # Stream response
    messages = [
        SystemMessage(content="You are a helpful assistant answering questions based on provided context."),
        HumanMessage(content=formatted_prompt)
    ]
    
    response = ""
    for chunk in llm.stream(messages):
        response += chunk.content
        yield chunk.content
    yield f"\n\n**Source Documents:**\n{docs}"

def main():
    st.title("Live RAG Chatbot: Upload PDF and Ask Questions")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'vectorstore_key' not in st.session_state:
        st.session_state.vectorstore_key = None

    # Toggle for display mode
    display_mode = st.radio("Response Display Mode", ["Stream (Word-by-Word)", "Pause and Display All"], index=0)

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        # Update session state with new file
        st.session_state.uploaded_file = uploaded_file
        # Use file name as a key to trigger vectorstore update
        st.session_state.vectorstore_key = uploaded_file.name

    # Load vectorstore
    vectorstore = get_vectorstore(uploaded_file_key=st.session_state.vectorstore_key)

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Chat input
    prompt = st.chat_input("Ask a question about the document or say something casual!")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Handle clear chat command
        if is_clear_chat(prompt):
            st.session_state.messages = []
            reply = "Chat history cleared! How may I help you now?"
            st.chat_message('assistant').markdown(reply)
            st.session_state.messages.append({'role': 'assistant', 'content': reply})
            return

        # Handle small talk
        if is_small_talk(prompt):
            reply = "How may I help you today? 😊 Upload a PDF or ask a question about the document!"
            st.chat_message('assistant').markdown(reply)
            st.session_state.messages.append({'role': 'assistant', 'content': reply})
            return

        # Process query
        try:
            if vectorstore is None:
                st.error("Please upload a PDF or ensure the FAISS database is available.")
                return

            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=0.0,
                groq_api_key=os.environ["GROQ_API_KEY"],
            )
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            prompt_template = set_custom_prompt()

            if display_mode == "Stream (Word-by-Word)":
                # Stream response
                placeholder = st.chat_message('assistant').empty()
                full_response = ""
                for token in stream_response(llm, retriever, prompt, prompt_template):
                    full_response += token
                    placeholder.markdown(full_response)
                st.session_state.messages.append({'role': 'assistant', 'content': full_response})
            else:
                # Pause and display all
                with st.spinner("Generating response..."):
                    time.sleep(1.5)  # Pause for 1.5 seconds
                    docs = retriever.invoke(prompt)
                    context = "\n".join([doc.page_content for doc in docs[:3]]) if docs else "No relevant information found."
                    formatted_prompt = prompt_template.format(context=context, question=prompt)
                    messages = [
                        SystemMessage(content="You are a helpful assistant answering questions based on provided context."),
                        HumanMessage(content=formatted_prompt)
                    ]
                    response = llm.invoke(messages).content
                    result_to_show = f"{response}\n\n**Source Documents:**\n{docs}"
                    st.chat_message('assistant').markdown(result_to_show)
                    st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()