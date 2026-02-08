import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

DB_PATH = "db"

st.title("📄 Chat with your Documents (Mini RAG)")

question = st.text_input("Ask a question about the PDF")

if question:

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # retrieve relevant chunks
    docs = db.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = OllamaLLM(model="llama3")

    prompt = f"""
You are an AI assistant answering questions using ONLY the provided context.

If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response)

    st.subheader("Sources")
    for doc in docs:
        st.write(doc.metadata)
