from fastapi import FastAPI
from pydantic import BaseModel
from .vectorstore import create_vectorstore
from .qa_system import create_qa



app = FastAPI(title="PDF QA API")

PDF_PATH = "/home/junaid/RAG_Task/Attention Is All You Need.pdf"
VECTORSTORE_DIR = "chroma_db"

# Initialize vectorstore and QA pipeline once
vectorstore = create_vectorstore(PDF_PATH, VECTORSTORE_DIR)
qa = create_qa(vectorstore)

class Query(BaseModel):
    question: str

@app.post("/ask/")
def ask_question(query: Query):
    answer = qa.run(query.question)
    return {"question": query.question, "answer": answer}
