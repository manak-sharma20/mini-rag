# Mini RAG Chatbot (Document Question Answering)

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about PDF documents.
The system indexes documents using embeddings and retrieves relevant context to generate grounded answers using a locally hosted LLaMA-3 model via Ollama.

------------------------------------------------------------

## Features
- Chat with PDF documents
- Semantic search (meaning-based, not keyword-based)
- Local LLM (no paid API required)
- Page-level citations
- Reduced hallucinations using grounded context

------------------------------------------------------------

## Tech Stack
- Python
- LangChain
- ChromaDB (Vector Database)
- Sentence Transformers (Embeddings)
- Ollama + LLaMA3 (Local LLM)
- Streamlit (User Interface)

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
- Multi-document upload
- Chat history memory
- PDF highlighting
- Cloud deployment

------------------------------------------------------------

## Author
Manak Sharma

AI/ML Project – Retrieval-Augmented Generation Chatbot
