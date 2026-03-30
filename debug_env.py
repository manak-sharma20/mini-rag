import os
from dotenv import load_dotenv

print(f"Current working directory: {os.getcwd()}")
print(f"Is .env file present? {os.path.exists('.env')}")

loaded = load_dotenv()
print(f"load_dotenv() returned: {loaded}")

api_key = os.environ.get('GROQ_API_KEY', 'NOT_FOUND')
print(f"GROQ_API_KEY: {api_key}")
