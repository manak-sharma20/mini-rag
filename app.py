import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DB_PATH = "db"


groq_api_key = os.environ.get("GROQ_API_KEY", "")

st.title("📄 Chat with your Documents (Mini RAG)")


with st.sidebar:
    st.header("Upload Documents")
    if groq_api_key:
        st.success("✅ Groq API key loaded from .env")
    else:
        st.error("❌ GROQ_API_KEY not found in .env — answers will be raw chunks")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if st.button("Process Uploaded PDFs") and uploaded_files:
        with st.spinner("Processing documents..."):
            documents = []
            # Save uploaded files to a temporary directory and load them
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                
                try:
                    loader = PyPDFLoader(temp_file_path)
                    docs = loader.load()
                    for d in docs:
                        # Keep original filename in metadata
                        d.metadata["source"] = uploaded_file.name
                    documents.extend(docs)
                finally:
                    os.unlink(temp_file_path)
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )
                chunks = text_splitter.split_documents(documents)
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                
                # Add directly to Chroma index
                db = Chroma.from_documents(
                    chunks,
                    embeddings,
                    persist_directory=DB_PATH
                )
                st.success(f"Successfully processed {len(uploaded_files)} PDF(s) and added {len(chunks)} chunks to the database.")
            else:
                st.warning("No text could be extracted from the uploaded PDFs.")

question = st.text_input("Ask a question about the PDF")

if question:
   
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    
    retriever = db.as_retriever(search_kwargs={"k": 4})

    if groq_api_key:
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        with st.spinner("Generating answer..."):
            try:
                response = rag_chain.invoke({"input": question})
                st.subheader("Answer")
                st.write(response["answer"])
                
                with st.expander("View Retrieved Context"):
                    for i, doc in enumerate(response["context"]):
                        source = doc.metadata.get('source', 'Unknown Document')
                        page = doc.metadata.get('page', 'Unknown Page')
                        st.markdown(f"**Source {i+1}:** {source} (Page {page})")
                        st.write(doc.page_content)
                        st.divider()
            except Exception as e:
                st.error(f"Error communicating with Groq API: {e}")
                
    else:
       
        st.warning("⚠️ No Groq API Key provided. Falling back to returning raw document chunks instead of LLaMA-generated answers. Please enter your API key in the sidebar.")
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        st.subheader("Retrieved Context (Evidence from Document)")
        st.write(context)
        
        st.subheader("Sources")
        for doc in docs:
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            st.write(f"Source: {source}, Page: {page}")
