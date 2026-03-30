import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

def verify():
    print("--- RAG Setup Verification ---")
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("❌ FAILED: GROQ_API_KEY not found in environment.")
        return
    
    print(f"✅ Key detected: {api_key[:10]}... (length: {len(api_key)})")
    
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
        response = llm.invoke("Hello, are you working?")
        print(f"✅ Success! LLM Response: {response.content[:50]}...")
    except Exception as e:
        print(f"❌ FAILED: API connection error: {e}")

if __name__ == "__main__":
    verify()
