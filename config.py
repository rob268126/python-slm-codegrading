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

# Dataset
KAGGLE_DATASET_HANDLE = os.getenv("KAGGLE_DATASET_HANDLE", "robviet/cpp-grading-dataset-public")
DATASET_ROOT = os.getenv("DATASET_ROOT", "./dataset")

# Kaggle API
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Kaggle Account
if KAGGLE_USERNAME and KAGGLE_KEY:
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY