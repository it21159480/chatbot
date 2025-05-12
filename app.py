import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')  # Get the Groq API key from the environment variable

# Load documents (modify path as needed)
documents = []
folder_path = "./content"  # This is the relative path to the 'content' folder # Modify this path accordingly

# Process files
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.lower().endswith((".ppt", ".pptx")):
        loader = UnstructuredPowerPointLoader(file_path)
        documents.extend(loader.load())

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create vector store - using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(texts, embeddings)

# Set up Groq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192", temperature=0)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=True
)

# Streamlit interface for asking questions
def ask_question(question):
    result = qa_chain({"query": question})
    st.write("Answer:", result["result"])
    st.write("\nSources:")
    for doc in result["source_documents"]:
        st.write(doc.metadata["source"], "- Page", doc.metadata.get("page", "N/A"))




# Function to handle previous Q&A storage
def ask_question(question):
    result = qa_chain({"query": question})
    
    # Store the question and answer in session state
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    
    # Add new question and answer to the history
    st.session_state.qa_history.append({"question": question, "answer": result["result"]})
    
    # Display the answer and sources
    st.write("Answer:", result["result"])
    st.write("\nSources:")
    for doc in result["source_documents"]:
        st.write(doc.metadata["source"], "- Page", doc.metadata.get("page", "N/A"))

# Streamlit UI
st.title("CTSE_LLM_Chatbot_Assignment")

# User input for question (text box)
question = st.text_area("Type your question here:")

# Add a Send button
send_button = st.button("Send")

# If Send button is clicked, process the question
if send_button and question:
    ask_question(question)

# Display the history of previous questions and answers
if "qa_history" in st.session_state:
    st.subheader("Previous Questions and Answers")
    for qa in st.session_state.qa_history:
        st.write(f"**Q**: {qa['question']}")
        st.write(f"**A**: {qa['answer']}")
        st.write("---")
