import os
import streamlit as st
from dotenv import load_dotenv
import weaviate
from langchain.prompts import ChatPromptTemplate #The line from langchain.prompts import ChatPromptTemplate imports the ChatPromptTemplate class from the prompts module within the langchain library. This class is specifically designed for creating templates used in chat-oriented question-answering tasks.
from langchain.schema.runnable import RunnablePassthrough #This line imports the RunnablePassthrough class from the runnable module within the schema subpackage of LangChain.is a basic component used in LangChain workflows. It simply passes the input data through the chain unchanged
from langchain.schema.output_parser import StrOutputParser #This line imports the StrOutputParser class from the output_parser module within the schema subpackage of LangChain.The StrOutputParser class is a component used for parsing the output of a LangChain workflow. In this case, it specifically expects the output to be a string. It essentially converts the output (which could be in various formats depending on the chain) into a simple string representation.
from langchain import HuggingFaceHub
from functions import load_pdf, split_text, generate_embeddings, save_to_weaviate,create_huggingface_model

# Load environment variables from the .env file
load_dotenv()

# Fetch the Weaviate URL and API key from environment variables
WEAVIATE_URL = os.getenv('CLUSTER_URL')
WEAVIATE_API_KEY = os.getenv('API_KEY')
huggingfacehub_api_token = os.getenv('huggingfacehub_api_token')
repo_id = os.getenv('repo_id')

# Ensure environment variables are loaded correctly
if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("WEAVIATE_URL or WEAVIATE_API_KEY not set in environment variables.")

# Set up Weaviate client
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
)

# Streamlit interface for uploading PDF
st.title("RAG-Based Chat Bot 🤖")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Step 1: Load PDF content
    pdf_content = load_pdf(uploaded_file)
    st.write("PDF loaded successfully!")
    
    # Step 2: Split the text into chunks
    documents = split_text(pdf_content)
    st.write(f"Text split into {len(documents)} chunks.")
    
    # Step 3: Generate embeddings for the documents
    embeddings = generate_embeddings()  # Pass documents to the function
    st.write("Embeddings generated for the documents.")

    # Step 4: Save the vectorized documents to Weaviate
    vector_db = save_to_weaviate(documents, embeddings, client)
    st.success("Documents are vectorized and saved to Weaviate.")
    
    # Step 5: Create the HuggingFace model
    st.write("Attempting to create the HuggingFace model...")
    model = create_huggingface_model(huggingfacehub_api_token=huggingfacehub_api_token, repo_id=repo_id)
    st.success("Model created successfully!")
    
    # Initialize retriever and RAG pipeline
    output_parser = StrOutputParser()
    retriever = vector_db.as_retriever()
    template = """
    You are an assistant for question-answering tasks.
    Use pieces of retrieved context and provide enhanced answer to the question.
    If there is no context,just respond with: 'Sorry, I didn’t understand your question. Would you like to connect with a live agent?'
    Question: {question}
    Context: {context}
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the RAG pipeline
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | output_parser
    )
    st.success("RAG pipeline created successfully!")
    
    # User Input for Questions
    question = st.text_input("Ask a question about the uploaded PDF:")

    def process_rag_response(question):
        """
        Function to invoke the RAG pipeline and process the response.
        It checks if the response contains the expected answer format.
        """
        try:
            # Invoke the RAG pipeline directly with the question
            response = rag_chain.invoke(question)

            # Ensure response is a string
            if not isinstance(response, str):
                response = str(response)

            # Check if the response contains "Answer:" and extract it
            if "Answer:" in response:
                answer = response.split("Answer:", 1)[1].strip()  # Split and extract the answer
                return answer
            else:
                return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"

        except Exception as e:
            # Handle any exceptions that occur
            import traceback
            st.error(f"An error occurred: {e}")
            print(traceback.format_exc())  # Log the error for debugging
            return "Sorry, there was an error processing your request."

    # Handle user input and display the answer
    if question:
        with st.spinner("Processing your question..."):
            answer = process_rag_response(question)
            st.write("### Answer:")
            st.write(answer)

else:
    st.warning("Please upload a PDF file to proceed.")
