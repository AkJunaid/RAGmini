import torch
import os
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read HuggingFace token from environment (optional)
hf_token = os.getenv("HF_TOKEN")

print("Loading BGE-M3 model for embeddings...")
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModel.from_pretrained(model_name, token=hf_token)
model.to(device)
model.eval()

def embed_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

class BGEEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [embed_text(t) for t in texts]

    def embed_query(self, text):
        return embed_text(text)
