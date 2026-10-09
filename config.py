import os
from dotenv import load_dotenv

load_dotenv()

# API Key cho Jev Model (TypeSafe AI / OpenRouter / Vercel Gateway)
JEV_API_KEY = os.getenv("JEV_API_KEY", "your_api_key_here")
JEV_BASE_URL = os.getenv("JEV_BASE_URL", "https://api.typesafe.ai/v1/jev") # Hoặc OpenRouter endpoint

# Model paths
BASE_MODEL_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_ADAPTER_PATH = "./models/lora_adapter_v3"