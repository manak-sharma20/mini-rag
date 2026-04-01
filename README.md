# RAG Intelligent Document Assistant 🔍

A premium, multi-model Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **LangChain**, and **Groq**. This application allows you to upload multiple PDF documents and engage in a high-speed, fact-grounded conversation with your data using state-of-the-art open-source LLMs.

---

## 🚀 Key Features

- **🧠 Multi-Model Intelligence**: Choose between several industry-leading models including LLaMA 3.3 (70B), LLaMA 3.1 (8B), Qwen, and GPT-OSS.
- **📁 Multi-PDF Context**: Upload and index multiple documents simultaneously for cross-document analysis.
- **🕒 Session-Based memory**: Full chat history with a sidebar for quick reference to previous queries.
- **🎯 Page-Level Citations**: Every answer includes expandable sections showing the exact source text and page numbers used.
- **⚡ High-Speed Inference**: Powered by **Groq Cloud** for sub-second responses.
- **🛡️ Production Hardened**: Optimized for both local Mac (Apple Silicon) and hosted environments (Streamlit Cloud).

---

## 🤖 Model Selection Guide

Compare and choose the best model for your specific task:

| Model | Classification | Best For... |
| :--- | :--- | :--- |
| **LLaMA 3.3 (70B)** | 🔥 **Powerhouse** | Complex reasoning, deep analysis, and high-accuracy specialized tasks. |
| **LLaMA 3.1 (8B)** | ⚡ **Speedster** | Quick summaries, basic factual queries, and lightning-fast chat response. |
| **Qwen 3 (32B)** | ⚖️ **All-Rounder** | Excellent balance of logic, coding help, and reasoning speed. |
| **GPT-OSS (120B)** | 🐘 **Heavyweight** | Extremely large context windows and the most demanding reasoning tasks. |

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit Dashboard
- **LLM API**: Groq Cloud (LLaMA 3 series)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Vector Store**: ChromaDB (locally persistent)
- **PDF Core**: PyPDF & LangChain Document Loaders

---

## ⚙️ Installation & Local Setup

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/manak-sharma20/mini-rag
   cd mini-rag
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   HUGGINGFACE_API_TOKEN=hf_your_token_here (Optional for local, required for Inference API)
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the App**:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment (Streamlit Cloud)

To deploy this project to the Streamlit Community Cloud:
1. Connect your GitHub repository to [share.streamlit.io](https://share.streamlit.io).
2. Add your `GROQ_API_KEY` to the **Secrets** section in the Streamlit Cloud dashboard.
3. The app includes built-in stability patches (SQLite3 monkeypatching and torch meta-tensor fixes) to ensure smooth operation on cloud nodes.

---

## 👨‍💻 Author
**Manak Sharma**

*Built as a high-performance demonstration of Retrieval-Augmented Generation.*
