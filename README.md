## PDF QA API with LangChain, BGE-M3 Embeddings & FastAPI

This project is a PDF Question-Answering API built with FastAPI, LangChain, Hugging Face Transformers, and Chroma vector store. It uses the BGE-M3 model for embeddings and FLAN-T5 for generating answers.

Project Structure


pdf_qa_project/

├─ app/ 

│  ├─ __init__.py          # Marks the app directory as a Python package 

│  ├─ main.py              # FastAPI entry point and API routes 

│  ├─ qa_system.py         # RetrievalQA pipeline (LLM + retriever) 

│  ├─ embeddings.py        # BGE-M3 embedding implementation 

│  └─ vectorstore.py       # PDF loading, chunking, and Chroma vector store 

├─ Attention Is All You Need.pdf   # Source PDF document 

├─ chroma_db/              # Persisted Chroma vector database (auto-generated) 

├─ requirements.txt        # Project dependencies 

└─ README.md               # Project documentation 



# Setup & Installation

Clone the repository

git clone <your-repo-url>
cd pdf_qa_project


Create a Python virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


# Install dependencies

pip install -r requirements.txt


Set Hugging Face token (optional but recommended for faster downloads)

export HF_TOKEN="your_huggingface_token_here"  # Linux/Mac
set HF_TOKEN=your_huggingface_token_here       # Windows

Running the API

Start the FastAPI server:

uvicorn app.main:app --reload


You will see output like:

Uvicorn running on http://127.0.0.1:8000

Usage

Send a POST request to the /ask/ endpoint:

curl -X POST http://127.0.0.1:8000/ask/ \
-H "Content-Type: application/json" \
-d '{"question": "What is the main idea of this paper?"}'


Response example:

{
    "question": "What is the main idea of this paper?",
    "answer": "The paper introduces the Transformer model, which uses self-attention mechanisms..."
}


You can also access the interactive Swagger UI at:

http://127.0.0.1:8000/docs

# Key Components
1. Embeddings

Uses BGE-M3 model for embedding PDF text chunks.

Embeddings are mean-pooled from the last hidden state.

Implemented as a LangChain Embeddings class in embeddings.py.

2. Vector Store

Uses Chroma to store embeddings.

Splits PDF into chunks using RecursiveCharacterTextSplitter.

Persisted on disk in chroma_db/ for re-use.

3. LLM for QA

Uses FLAN-T5-small via Hugging Face pipeline.

Wrapped in LangChain HuggingFacePipeline.

Combined with Chroma retriever in RetrievalQA.

4. FastAPI

Single endpoint /ask/ to query the PDF.

Handles JSON POST requests with a question field.

Returns the LLM-generated answer.
