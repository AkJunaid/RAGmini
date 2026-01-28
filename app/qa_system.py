import torch
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

def create_qa(vectorstore):
    device_hf = 0 if torch.cuda.is_available() else -1
    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=device_hf,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return qa