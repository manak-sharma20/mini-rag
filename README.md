# Mini RAG Chatbot (Document Question Answering)

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about PDF documents.
The system indexes documents using embeddings and retrieves relevant context to generate grounded answers using high-speed inference via **Groq** (LLaMA 3.3/3.1).

------------------------------------------------------------

## Features
- **Multi-Model Support**: Switch between LLaMA 3.3, 3.1, Qwen, and GPT-OSS models.
- **Chat History**: Full session-based conversation memory with a "Recent Chats" sidebar.
- **Multi-PDF Upload**: Upload and process multiple documents directly through the UI.
- **Semantic search**: Meaning-based retrieval using HuggingFace embeddings.
- **Page-level citations**: Verified sources and page numbers for every answer.
- **Mac Optimized**: Includes fixes for Torch meta-tensor errors on Apple Silicon.

------------------------------------------------------------

## Tech Stack
- **Python** & **LangChain**
- **ChromaDB** (Vector Database)
- **HuggingFace** (`all-MiniLM-L6-v2` Embeddings)
- **GroqCloud** (LLaMA 3.3/3.1 Inference)
- **Streamlit** (UI Framework)

------------------------------------------------------------

## Project Architecture

User Question → Embedding → Vector Search (ChromaDB) → Retrieve Context → LLaMA3 → Answer + Citations

This pipeline is called Retrieval-Augmented Generation (RAG).

------------------------------------------------------------

## How It Works

1. PDFs are loaded and converted into text.
2. Text is split into smaller chunks.
3. Each chunk is converted into embeddings (numerical meaning representation).
4. Embeddings are stored in a vector database (ChromaDB).
5. User asks a question.
6. System retrieves relevant chunks.
7. Context is passed to the LLM (LLaMA3).
8. The LLM generates a grounded answer with sources.

------------------------------------------------------------

## Installation

Clone the repository:

git clone <your-repo-url>
cd mini-rag

Create virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------

## Usage

1) Add your PDFs inside the data/ folder

2) Index the documents:

python ingest.py

3) Start the chatbot:

streamlit run app.py

Open browser:
http://localhost:8501

------------------------------------------------------------

## Example Questions

- What is self-attention?
- Explain transformer architecture
- What problem does the paper solve?

------------------------------------------------------------

## Why RAG Instead of Fine-Tuning?

Fine-tuning modifies model weights and is expensive and static.
RAG keeps the model unchanged and dynamically retrieves knowledge from documents, allowing:
- Easy updates
- Lower cost
- Factual answers
- Source citations

------------------------------------------------------------

## Future Improvements
- PDF highlighting in the viewer
- Cloud deployment (Streamlit Community Cloud)
- Long-term chat persistence (Database-backed)

------------------------------------------------------------

## Author
Manak Sharma

AI/ML Project – Retrieval-Augmented Generation Chatbot
