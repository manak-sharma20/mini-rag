import os
import torch

# Universal Stability Overrides (Fixes meta-tensor error)
os.environ["TRANSFORMERS_ACCELERATE_OFF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_default_device('cpu')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

DATA_PATH = "data"
DB_PATH = "db"

documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        docs = loader.load()
        documents.extend(docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'device': 'cpu'}
)

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=DB_PATH
)



print("Indexing complete!")
