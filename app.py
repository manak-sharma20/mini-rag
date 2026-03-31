import streamlit as st
import os
import torch

# Universal Stability Overrides (Fixes meta-tensor error)
os.environ["TRANSFORMERS_ACCELERATE_OFF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# Initialize session state for chat history early for sidebar access
if "messages" not in st.session_state:
    st.session_state.messages = []

DB_PATH = "db"


groq_api_key = os.environ.get("GROQ_API_KEY", "")

st.title("Chat with your Documents")


with st.sidebar:
    st.header("Settings")
    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found in .env")
    
    # Updated Model Selection (March 2026 Active List)
    model_option = st.selectbox(
        "Choose Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b", "openai/gpt-oss-120b"],
        index=1 # llama-3.1-8b-instant
    )
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("Recent Chats")
    for i, msg in enumerate(st.session_state.get("messages", [])):
        if msg["role"] == "user":
            # Show a truncated version of the user query in the sidebar
            if st.button(f"💬 {msg['content'][:30]}...", key=f"hist_{i}"):
                # This could be used to jump to a message, but for now just as a list
                pass

    st.divider()
    st.header("Upload Documents")
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
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'device': 'cpu'}
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

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("View Retrieved Context"):
                for i, doc in enumerate(message["context"]):
                    source = doc.get('source', 'Unknown Document')
                    page = doc.get('page', 'Unknown Page')
                    st.markdown(f"**Source {i+1}:** {source} (Page {page})")
                    st.write(doc.get('content', ''))
                    st.divider()

# React to user input
if prompt := st.chat_input("Ask a question about your documents"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'device': 'cpu'}
    )

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})

    with st.chat_message("assistant"):
        if groq_api_key:
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_option,
                temperature=0
            )
            
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Use three sentences maximum and keep the answer concise.\n\n"
                "{context}"
            )
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            with st.spinner("Generating answer..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Process context for storage
                    context_data = []
                    for doc in response["context"]:
                        context_data.append({
                            "source": doc.metadata.get('source', 'Unknown'),
                            "page": doc.metadata.get('page', 'Unknown'),
                            "content": doc.page_content
                        })
                    
                    with st.expander("View Retrieved Context"):
                        for i, cd in enumerate(context_data):
                            st.markdown(f"**Source {i+1}:** {cd['source']} (Page {cd['page']})")
                            st.write(cd['content'])
                            st.divider()
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "context": context_data
                    })
                except Exception as e:
                    st.error(f"Error communicating with Groq API: {e}")
        else:
            st.warning("⚠️ No Groq API Key provided. Returning raw document chunks.")
            docs = retriever.invoke(prompt)
            context = "\n\n".join([doc.page_content for doc in docs])
            st.markdown(context)
            
            context_data = []
            for doc in docs:
                context_data.append({
                    "source": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', 'Unknown'),
                    "content": doc.page_content
                })

            st.session_state.messages.append({
                "role": "assistant", 
                "content": context,
                "context": context_data
            })
