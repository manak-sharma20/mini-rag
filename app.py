import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


DB_PATH = "db"

st.title("📄 Chat with your Documents (Mini RAG)")

question = st.text_input("Ask a question about the PDF")

if question:

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # retrieve relevant chunks
    docs = db.similarity_search(question, k=4)

    context = "\n\n".join([doc.page_content for doc in docs])

    st.subheader("Retrieved Context (Evidence from Document)")
    st.write(context)

    st.info("Note: Full AI answering (LLaMA3) works locally. Cloud demo shows retrieved knowledge chunks only.")


    st.subheader("Sources")
    for doc in docs:
        st.write(doc.metadata)
