# functions.py
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain_community.vectorstores import Weaviate
import weaviate
import tempfile

def load_pdf(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())  # Write the file content
        temp_file_path = temp_file.name  # Get the temp file path
        
    # Now load the PDF using the temp file path
    loader = PyPDFLoader(temp_file_path, extract_images=True)
    return loader.load()

def split_text(content, chunk_size=1000, chunk_overlap=200):
    """
    Splits the extracted PDF content into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(content)

def generate_embeddings():
    """
    Generates embeddings for the given text using Hugging Face model.
    """
    model_name="sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def save_to_weaviate(docs, embeddings, client):
    """
    Saves the vectorized documents to Weaviate vector database.
    """
    vector_db = Weaviate.from_documents(
        docs,
        embeddings,
        client=client,
        by_text=False
    )
    return vector_db

# Function to initialize the HuggingFace model
def create_huggingface_model(huggingfacehub_api_token: str, repo_id: str, temperature: float = 1.0, max_length: int = 400):
    """
    Initializes and returns a HuggingFace model using the HuggingFaceHub API.
    
    Parameters:
        huggingfacehub_api_token (str): The HuggingFace API token for authentication.
        repo_id (str): The ID of the model repository on HuggingFace.
        temperature (float): Controls the randomness of responses. Defaults to 1.0.
        max_length (int): Maximum length of the output sequence. Defaults to 400.
        
    Returns:
        HuggingFaceHub: The initialized HuggingFaceHub model.
    """
    try:
        model = HuggingFaceHub(
            huggingfacehub_api_token=huggingfacehub_api_token,
            repo_id=repo_id,
            model_kwargs={"temperature": temperature, "max_length": max_length}
        )
        return model
    except Exception as e:
        print(f"Error in creating model: {e}")
        return None
