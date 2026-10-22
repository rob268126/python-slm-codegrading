"""
config.py
Cấu hình hệ thống Local JEV và đường dẫn mô hình.
Hệ thống sử dụng Qwen2.5-Coder-7B chạy cục bộ làm Local JEV Judge (không phụ thuộc external API).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Đường dẫn mô hình
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "unsloth/Qwen2.5-Coder-7B-Instruct")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./models/lora_adapter_v3")

# Tham số Local JEV Engine
JEV_TEMPERATURE = float(os.getenv("JEV_TEMPERATURE", "0.0"))
JEV_MAX_RETRIES = int(os.getenv("JEV_MAX_RETRIES", "1"))
JEV_VERIFIER_ENABLED = os.getenv("JEV_VERIFIER_ENABLED", "true").lower() == "true"