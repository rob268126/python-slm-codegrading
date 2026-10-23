"""
config.py
Cấu hình đường dẫn mô hình (Qwen2.5-Coder-7B + LoRA chạy cục bộ, không dùng external API).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Đường dẫn mô hình
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "unsloth/Qwen2.5-Coder-7B-Instruct")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./models/lora_adapter_v3")