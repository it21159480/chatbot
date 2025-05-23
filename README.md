# chatbot

# CTSE Lecture Notes Chatbot

This project involves developing a chatbot that can answer questions based on **CTSE lecture notes**. The chatbot uses **Large Language Models (LLMs)** integrated with the **LangChain** framework and **Groq API** to process and generate responses. The system uses **FAISS** for document retrieval and **HuggingFaceEmbeddings** for generating document embeddings, allowing the chatbot to provide accurate answers to user queries.

---

## Features

- **Retrieval-Augmented Generation (RAG)** approach to answer questions from CTSE lecture notes.
- **Groq API** for generating answers based on document retrieval.
- Supports **PDF** and **PowerPoint** input for lecture notes.
- **Streamlit UI** for easy interaction with the chatbot.
- **Local and cloud deployment** options available.

---

## Requirements

Before running this project, ensure you have the following:

- Python 3.x
- Streamlit (for deploying the web app)
- FAISS (for document retrieval)
- HuggingFace for embeddings

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/it21159480/chatbot.git
cd chatbot

```

### 2. Set Up Virtual Environment

```bash
python -m venv venv

venv\Scripts\activate

```
### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```
### 4. Create a .env file in the project folder with the following content

```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```
### 5. Run 

#### 1. locally 
```bash
python app.py
```
#### 2. Streamlit Application
``` bash
streamlit run app.py
```
