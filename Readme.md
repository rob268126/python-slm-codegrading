# C++ Code Grading & Feedback Pipeline (SLM)

Hệ thống chấm điểm và sinh phản hồi sư phạm tự động cho bài nộp C++ của sinh viên, kết hợp **Qwen2.5-Coder-7B + LoRA**, **g++ Compiler Sandbox** và **Differential Testing**.

---

## Tổng quan 3 Tasks

| Task | Mô tả | Đầu ra (`predictions.json`) | Metric chính |
| :--- | :--- | :--- | :--- |
| **Task 1** | Chấm điểm Rubric 6 chiều (`compilable`, `io_format`, `logic`, `edge_case`, `complexity`, `code_quality`) | `rubric` + `total_score` (0–10) | **QWK** (Quadratic Weighted Kappa), MAE |
| **Task 2** | Phân loại lỗi đa nhãn trên 10 nhãn Taxonomy chính thức | `taxonomy_error` | **Macro-F1**, Micro-F1 |
| **Task 3** | Sinh phản hồi tiếng Việt có kiểm soát mức độ sư phạm (Level 1 → 4) | `taxonomy_error` + `feedback` | **Level Compliance** & Độ chính xác chẩn đoán |

---

## Cấu trúc thư mục

```text
python-slm-codegrading/
├── main.py                      # Entry point chạy toàn bộ pipeline (Task 1, 2, 3)
├── config.py                    # Cấu hình đường dẫn base model & LoRA adapter
├── requirements.txt             # Thư viện Python cần thiết
├── core/
│   ├── schema.py                # Pydantic schema, chuẩn hóa 10 nhãn taxonomy & rubric
│   └── data.py                  # Loader dữ liệu (train / public_test / private_test)
├── sandbox/
│   ├── cpp_compiler.py          # g++ Compiler Sandbox (ground truth biên dịch + chạy code)
│   └── differential.py          # Sinh reference solution & test tự động cho single_problem
├── inference/
│   ├── llm.py                   # Wrapper Qwen2.5-Coder + LoRA & rút gọn prompt thông minh
│   ├── grader.py                # Chấm điểm Task 1 (Rubric) & Task 2 (Taxonomy)
│   └── feedback_gen.py          # Sinh phản hồi Task 3 bám bằng chứng thực thi theo Level
├── evaluation/
│   └── metrics.py               # Tính QWK, MAE, Macro-F1, Micro-F1, Level Compliance
├── notebook/
│   └── pipeline-cpp-grading-v5.ipynb  # Notebook end-to-end (Fine-tune + Inference trên Kaggle)
└── submission/                  # Thư mục xuất file nộp: submission/task{1,2,3}/predictions.json
```

---

## Hướng dẫn chạy

### 1. Chạy trên Kaggle (Khuyến nghị)
1. Mở notebook `notebook/pipeline-cpp-grading-v5.ipynb` trên Kaggle (GPU T4x2 hoặc P100).
2. Add dataset tại đường dẫn: `/kaggle/input/datasets/robviet/cpp-grading-dataset-public` (gồm 2 thư mục `train` và `public_test`).
3. Chạy toàn bộ notebook để **Fine-tune LoRA** $\rightarrow$ **Inference cả 3 tasks** $\rightarrow$ xuất `submission.zip`.

### 2. Chạy trên máy cá nhân (Local)

**Bước 1: Cài đặt môi trường & `g++`**
```bash
pip install -r requirements.txt
g++ --version  # Đảm bảo đã có g++ trong PATH (MSYS2 / MinGW trên Windows)
```

**Bước 2: Chạy Pipeline**
- **Chạy đầy đủ với LLM + LoRA (cần GPU):**
  ```bash
  python main.py --dataset ./dataset --split public_test --task all --adapter-path ./models/lora_adapter_v3
  ```
- **Chạy từng task riêng lẻ (`1`, `2`, hoặc `3`):**
  ```bash
  python main.py --dataset ./dataset --split public_test --task 1 --adapter-path ./models/lora_adapter_v3
  ```
- **Chạy nhanh chế độ Heuristic (không cần GPU):**
  ```bash
  python main.py --dataset ./dataset --split public_test --task all --heuristic
  ```

Kết quả được xuất tự động tại: `submission/task1/predictions.json`, `submission/task2/predictions.json`, và `submission/task3/predictions.json`.

