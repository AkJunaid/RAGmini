
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .embeddings import BGEEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

def create_vectorstore(pdf_path: str, persist_directory: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    embedding_model = BGEEmbeddings()
    
    vectorstore = Chroma.from_texts(
        texts=[doc.page_content for doc in docs],
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore
