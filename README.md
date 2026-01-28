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


## the big problem i faced and solved at 5:02pm
The issue arises from a dependency mismatch between the transformers, torch, and triton packages in the Python environment.

Root Cause

When AutoTokenizer or AutoModel is imported from the transformers library, it internally imports PyTorch (torch). The installed PyTorch version (2.10.0) was built with GPU/CUDA support, which introduces a dependency on the Triton library for GPU kernel compilation and optimization.

This leads to the following error chain:

transformers imports AutoTokenizer

This triggers loading of torch._inductor

torch._inductor attempts to import triton.backends

triton is either missing or incompatible

The import process fails with ModuleNotFoundError

Why This Happens

PyTorch 2.x introduced torch.compile() and the Inductor backend, which relies on Triton to generate optimized GPU kernels. Even if these features are not explicitly used, the presence of a GPU-enabled PyTorch build causes these modules to be imported during initialization.

On machines without a GPU, these GPU-specific components are unnecessary and unusable. However, pip installs the GPU-enabled PyTorch wheel by default, which leads to this incompatibility.

Notebook vs. Local Environment

The Jupyter notebook (one-page-rag.ipynb) was designed to run on Kaggle, which provides GPU resources. The code checks torch.cuda.is_available() and works correctly in that environment.

In the local setup, there is no GPU. The GPU-enabled PyTorch build fails during import time, before the CUDA availability check can run, causing the application to crash.

Current Environment State

There are multiple Python virtual environments (eduverse, RAG) with conflicting dependency versions:

transformers==4.57.1 (requires newer PyTorch internals)

torch==2.10.0 (GPU-enabled build on a CPU-only machine)

langchain-huggingface requires langchain-core>=1.2.0, but version 0.2.43 is installed

These version mismatches compound the problem.

The Solution

Install the CPU-only PyTorch build, which does not depend on Triton:

pip uninstall -y torch triton
pip install --index-url https://download.pytorch.org/whl/cpu torch


This installs a PyTorch version that:

Has no GPU or CUDA dependencies

Does not require Triton

Works correctly on CPU-only systems

Is smaller and more stable for development

Why CPU-Only PyTorch Works

The CPU-only PyTorch build is compiled without CUDA support. As a result:

torch._inductor does not attempt to load Triton

All tensor operations run purely on the CPU

Performance is slower but fully functional for development and testing

Summary

The application code is correct. The failure is caused by a runtime environment mismatch—specifically, using a GPU-enabled PyTorch build on a machine without GPU support. Installing the correct CPU-only PyTorch distribution resolves the issue completely.
