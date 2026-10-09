# LLM Chấm điểm C++
## Ý tưởng
Chấm điểm và đánh giá bài nộp lập trình C++ của sinh viên:    
- LLM Fine-tuned (Qwen2.5-Coder) với LoRA adapter để chấm điểm theo rubric chi tiết
- Compiler Sandbox (g++) để xác thực biên dịch thực tế, đảm bảo tính chính xác
- Ensemble Strategy kết hợp nhiều "judges" để tăng độ tin cậy
- Feedback Generator sinh phản hồi tiếng Việt theo 4 mức độ kiểm soát (từ gợi ý nhẹ đến lời giải đầy đủ)
3 tasks chính:  
- Task 1: Rubric Grading  
Chấm điểm theo 6 tiêu chí (compilable, io_format, logic, edge_case, complexity, code_quality)  
- Task 2: Error Classification  
Phân loại lỗi theo taxonomy (Lỗi biên dịch, logic, vòng lặp, hàm, I/O, bộ nhớ, ...)  
- Task 3: Feedback Generation  
Sinh phản hồi sư phạm có kiểm soát mức độ chi tiết  

## Thư mục
prequisition/  
├── README.md                    # File hướng dẫn này  
├── requirements.txt             # Các dependencies Python  
├── main.py                      # Pipeline chính điều phối toàn bộ flow  
├── dataset/                     # Thư mục chứa dataset (input)  
│   ├── task1_grading.json  
│   ├── task2_error_taxonomy.json  
│   ├── task3_feedback.json  
│   ├── exams.json  
│   └── label_space.json  
├── models/                      # Thư mục chứa model weights  
│   └── lora_adapter_v3/         # LoRA adapter đã fine-tune  
├── evaluation/  
│   └── metrics.py               # Các hàm tính toán metric đánh giá  
├── inference/  
│   ├── grader.py                # Module chấm điểm chính (Task 1 + Task 2)  
│   ├── ensemble.py              # Ensemble chiến lược kết hợp nhiều judges  
│   └── feedback_gen.py          # Module sinh phản hồi (Task 3)  
├── sandbox/  
│   └── compiler.py              # C++ Compiler Sandbox (ground truth)  
└── predictions.json             # Output predictions sau khi chạy  

## Presiquisition (Compile C)
g++ (Windows) for compile C
```bash
winget install MSYS2.MSYS2
C:\msys64\ucrt64.exe
pacman -Syu
pacman -S mingw-w64-ucrt-x86_64-gcc
```
Add env path
```bash
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -notlike "*C:\msys64\ucrt64\bin*") {
    [Environment]::SetEnvironmentVariable(
        "Path",
        "$userPath;C:\msys64\ucrt64\bin",
        "User"
    )
}
```
Check environment
```bash
g++ --version
```

## Output models
Link: https://www.kaggle.com/code/robviet/pipeline-cpp-grading-v5
lora_adapter_v3.zip -> models/

## Setup + Run
```bash 
pip install -r requirements.txt
python main.py --dataset ./dataset/sample_dataset --model-path unsloth/Qwen2.5-Coder-7B-Instruct --adapter-path ./models/lora_adapter_v3 --output predictions.json
``` 

## Docs
### 1. main.py -> chạy Pipeline
💡 Ý tưởng:  
Đây là entry point của toàn bộ hệ thống. File điều phối flow: load dataset → compile code → inference bằng LLM → evaluate → lưu kết quả.  
🔄 Các bước chạy:  
Load dataset: Đọc các file JSON từ thư mục dataset/ (task1, task2, task3, exams, label_space)  
Load model: Load base model Qwen2.5-Coder + LoRA adapter (nếu có)  
Compile code: Với mỗi sample, gọi compile_cpp() để lấy kết quả biên dịch thực tế làm ground truth  
Tạo synthetic labels: Nếu không có error labels từ dataset, tạo từ compiler output và rubric  
Inference: Build prompt (ChatML format) và gọi LLM để chấm điểm  
Parse output: Extract JSON từ response của model (có fallback nếu model output không chuẩn)  
Evaluate: Tính QWK, MAE (Task 1), macro-F1 (Task 2)  
Save predictions: Lưu kết quả ra file predictions.json  
🚀 Cách chạy:  
```bash
python main.py \
    --dataset ./dataset \
    --model-path unsloth/Qwen2.5-Coder-7B-Instruct \
    --adapter-path ./models/lora_adapter_v3 \
    --output predictions.json
```

### 2. evaluation/metrics.py - Các hàm tính metric
💡 Ý tưởng:  
Đánh giá chất lượng của hệ thống so với ground truth. Dùng các metric chuẩn của ML cho từng task.  
🔄 Các hàm chính:  
quadratic_weighted_kappa(y_true, y_pred)  
-> Task 1 - Đo độ đồng thuận giữa prediction và ground truth trên tổng điểm  
-> Cohen's Kappa với weights="quadratic"  
evaluate_all(predictions, task1, task2, task3)  
-> Chạy toàn bộ evaluation cho 3 tasks  
-> QWK + MAE + Macro-F1 + Level Compliance  
  
📊 Các metric được tính:  
Task 1 (Rubric):  
QWK (Quadratic Weighted Kappa): Metric chính, đo độ đồng thuận. Giá trị càng gần 1.0 càng tốt.  
MAE (Mean Absolute Error): Sai số trung bình tuyệt đối giữa tổng điểm prediction và ground truth.    
Task 2 (Error Classification):   
Macro-F1: F1-score trung bình trên tất cả các nhãn lỗi (multi-label classification).   
Task 3 (Feedback):  
Level Compliance Rate: Tỷ lệ feedback tuân thủ đúng mức độ kiểm soát (từ Level 1 → 4).  


### 3. inference/grader.py - Module chấm điểm (Task 1 + Task 2)
💡 Ý tưởng:  
Đây là "trái tim" của hệ thống. Sử dụng LLM fine-tuned để chấm điểm rubric và phân loại lỗi, đồng thời có rule-based correction dựa trên kết quả compiler thực tế để đảm bảo tính chính xác.  
🔄 Các bước chạy:  
_load_model(): Load base model Qwen2.5 + LoRA adapter (với 4-bit quantization để tiết kiệm VRAM)  
build_prompt(): Xây dựng prompt theo format ChatML, bao gồm đề bài, code, compile log, test report
grade():  
Gọi LLM để generate response (greedy decoding với temperature=0.0 để deterministic)  
Parse JSON từ response (có fallback regex)  
Quan trọng: Áp dụng _apply_compiler_rules() để sửa rubric dựa trên compiler  
_apply_compiler_rules(): Rule cứng:  
Nếu code KHÔNG biên dịch được → compilable=0, logic=0, edge_case=0, complexity=0  
Nếu code biên dịch được → compilable=1  
Clamp các giá trị về đúng miền (ví dụ: logic 0-4, edge_case 0-2)  
grade_batch(): Chấm điểm hàng loạt nhiều submissions  

Rubric 6 tiêu chí:
compilable (0-1) -> Code có biên dịch được không?  
io_format (0-1) -> Format input/output có đúng không?  
logic (0-4) -> Thuật toán có đúng không?  
edge_case (0-2) -> Có xử lý các trường hợp biên không?  
complexity (0-1) -> Độ phức tạp thời gian/không gian có phù hợp không?  
code_quality (0-1) -> Code có sạch, dễ đọc không?  


### 5. inference/feedback_gen.py - Feedback Generator (Task 3)
💡 Ý tưởng:  
Sinh phản hồi tiếng Việt cho sinh viên với 4 mức độ kiểm soát:  
- Level 1: Gợi ý nhẹ nhàng, hướng tư duy. KHÔNG code, KHÔNG lời giải  
- Level 2: Chỉ ra vị trí lỗi, KHÔNG code sửa hoàn chỉnh  
- Level 3: Giải thích nguyên nhân + cách khắc phục, có thể có snippet nhỏ  
- Level 4: Cung cấp lời giải mẫu hoàn chỉnh  
🔄 Các bước chạy:  
generate(code, statement, errors, level, max_retries=2):  
- Gọi LLM để sinh feedback  
- Kiểm tra level compliance bằng JEV Noul (nếu có) hoặc heuristic  
- Nếu vi phạm → retry với prompt nghiêm khắc hơn (tối đa 2 lần)  
- Nếu vẫn vi phạm → strip code blocks khỏi feedback  
_generate_once(): Sinh feedback 1 lần với prompt cụ thể cho level  
_heuristic_check(): Kiểm tra nhanh bằng keyword (ví dụ: #include, int main(), cout)  
_strip_code(): Xóa các code blocks để đảm bảo compliance (dùng regex)  
📊 Output:   
```bash
{
    "feedback": "Bài làm tốt...",
    "level": 2,
    "level_compliant": true,
    "compliance_score": 0.95,
    "attempts": 1
}
```

### 6. sandbox/compiler.py - C++ Compiler Sandbox
💡 Ý tưởng:  
Cung cấp ground truth từ kết quả biên dịch thực tế bằng g++. Kiểm tra tính đúng đắn của code, giúp hệ thống không bị hallucination của LLM ảnh hưởng.   
🔄 Các bước chạy:  
find_gcc(): Tự động tìm đường dẫn g++ trên hệ thống:  
Tìm trong các đường dẫn phổ biến trên Windows (MSYS2, MinGW, TDM-GCC, ...)  
compile_cpp(code, timeout=15):  
Tạo file .cpp tạm thời  
Gọi g++ -std=c++17 -O2 -Wall -o output.exe input.cpp  
Parse stderr để phân loại lỗi (cú pháp, kiểu dữ liệu, undeclared, ...)  
Trả về dict với compilable, compile_success, compile_log, errors, warnings, error_types  
run_cpp_with_input(code, test_input, timeout=5):  
Compile code  
Chạy file .exe với input cho trước  
Trả về runtime_output, runtime_error, runtime_success, timed_out  
📋 Các loại lỗi được phân loại:  
Lỗi biến/hàm chưa khai báo (undeclared, not declared)  
Lỗi hàm (no match for, no matching function)  
Lỗi cú pháp (expected ;, expected }, ...)  
Lỗi kiểu dữ liệu (cannot convert, invalid conversion)  
Lỗi khai báo trùng (redeclaration, redefinition)  
Lỗi liên kết (undefined reference, ld returned)  
Lỗi thiếu header file (no such file)  
