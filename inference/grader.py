"""
inference/grader.py
Task 1 + Task 2: Chấm điểm rubric + phân loại lỗi
"""
import json
import torch
import os
import re  # <--- ĐÃ THÊM IMPORT RE
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

@dataclass
class GradingResult:
    sample_id: str
    rubric: Dict[str, int]
    errors: List[str]
    rationale: str
    compiler_verified: bool = False

class CppGrader:
    """Chấm điểm C++ submissions bằng fine-tuned LLM."""

    RUBRIC_DIMS = ["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"]

    RUBRIC_RANGES = {
        "compilable": (0, 1), "io_format": (0, 1), "logic": (0, 4),
        "edge_case": (0, 2), "complexity": (0, 1), "code_quality": (0, 1)
    }
    
    OFFICIAL_LABELS = [
        "Lỗi biên dịch", "Lỗi logic", "Lỗi vòng lặp", "Lỗi hàm", "Lỗi I/O",
        "Lỗi bộ nhớ", "Lỗi kiểu dữ liệu", "Lỗi thuật toán", "Lỗi xử lý biên", "Lỗi cú pháp"
    ]

    LABEL_MAP = {
        "Không có lỗi": None, "No error": None,
        "Lỗi biến/hàm chưa khai báo": "Lỗi cú pháp",
        "Lỗi liên kết (missing main)": "Lỗi biên dịch",
        "Lỗi biên dịch khác": "Lỗi biên dịch",
        "Lỗi nhập/xuất": "Lỗi I/O",
        "Lỗi edge case": "Lỗi xử lý biên",
        "Lỗi mảng/chuỗi": "Lỗi logic",
        "Lỗi hard-code": "Lỗi logic",
        "Lỗi style": None,
    }

    def __init__(self, base_model: str, adapter_path: str = None, device: str = "cuda"):
        self.device = device
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
        if not self.adapter_path or not os.path.exists(os.path.join(self.adapter_path, "adapter_config.json")):
            raise FileNotFoundError(f"LỖI: Không tìm thấy adapter hợp lệ tại {self.adapter_path}. Hãy kiểm tra lại đường dẫn!")
            
        self._load_model()

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
        from peft import PeftModel

        print(f"Loading base model: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Load LoRA adapter from {self.adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model = self.model.merge_and_unload()
        self.model.eval()

        self.gen_config = GenerationConfig(
            temperature=0.0, top_p=1.0, do_sample=False,
            max_new_tokens=1024, pad_token_id=self.tokenizer.eos_token_id,
        )

    def build_prompt(self, statement: str, code: str, compile_log: str,
                    test_report: str = "") -> str:
        """Xây dựng prompt ChatML format - PHẢI KHỚP với training prompt."""
        compile_status = "COMPILE SUCCESS" if "error" not in compile_log.lower() else "COMPILE FAILED"
        
        if not compile_log:
            compile_log_display = "Không có lỗi biên dịch"
        else:
            compile_log_display = compile_log[:500]

        input_text = (
            f"### Đề bài:\n{statement}\n\n"
            f"### Code sinh viên:\n```cpp\n{code}\n```\n\n"
            f"### Kết quả biên dịch thực tế (g++ -std=c++17):\n{compile_status}\n"
            f"Log: {compile_log_display}\n\n"
            f"### Test Report:\n{test_report if test_report else 'Không có'}"
        )

        prompt = (
            f"<|im_start|>system\n"
            f"You are an expert C++ code grader acting as a JUDGE. "
            f"You have ACTUAL COMPILATION RESULTS as ground truth - trust them above all.\n"
            f"Grading rubric:\n"
            f"- compilable (0-1): Does code compile? Use compiler result.\n"
            f"- io_format (0-1): Correct input/output format?\n"
            f"- logic (0-4): Algorithm correctness. IF compilable=0, this MUST be 0.\n"
            f"- edge_case (0-2): Handles edge cases? IF compilable=0, this MUST be 0.\n"
            f"- complexity (0-1): Appropriate time/space complexity?\n"
            f"- code_quality (0-1): Clean, readable code?\n\n"
            f"Error taxonomy (multi-label): Lỗi biên dịch, Lỗi logic, Lỗi vòng lặp, "
            f"Lỗi hàm, Lỗi I/O, Lỗi bộ nhớ, Lỗi kiểu dữ liệu, Lỗi thuật toán, "
            f"Lỗi xử lý biên, Lỗi cú pháp, Không có lỗi\n\n"
            f"Output ONLY valid JSON: {{\"rubric\": {{...}}, \"errors\": [...]}}<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"Rationale: \n"
        )
        return prompt

    def grade(self, statement: str, code: str, compile_log: str, test_report: str = "") -> GradingResult:
        prompt = self.build_prompt(statement, code, compile_log, test_report)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=self.gen_config)

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        rubric, errors, rationale = self._parse_response(response, compile_log, test_report, code)
        rubric = self._apply_compiler_rules(rubric, compile_log)

        return GradingResult(
            sample_id="", rubric=rubric, errors=errors,
            rationale=rationale, compiler_verified=True,
        )

    def _apply_compiler_rules(self, rubric: dict, compile_log: str) -> dict:
        has_error = "error" in compile_log.lower()
        if has_error:
            rubric["compilable"] = 0
            rubric["logic"] = 0
            rubric["edge_case"] = 0
            rubric["complexity"] = 0
        else:
            rubric["compilable"] = 1

        for dim, (lo, hi) in self.RUBRIC_RANGES.items():
            rubric[dim] = max(lo, min(hi, rubric.get(dim, 0)))
        return rubric

    def heuristic_rubric(self, compile_log: str = "", test_report=None, code: str = "") -> dict:
        compile_log = (compile_log or "").lower()
        if not code or len(code.strip()) == 0 or "error" in compile_log:
            return {d: 0 for d in self.RUBRIC_DIMS}

        rubric = {"compilable": 1, "io_format": 1, "logic": 2, "edge_case": 1, "complexity": 1, "code_quality": 1}
        if isinstance(test_report, list) and len(test_report) > 0:
            passed = sum(1 for t in test_report if t.get("passed", False))
            rate = passed / len(test_report)
            if rate == 1.0: rubric["logic"], rubric["edge_case"] = 4, 2
            elif rate >= 0.75: rubric["logic"], rubric["edge_case"] = 3, 2
            elif rate >= 0.5: rubric["logic"], rubric["edge_case"] = 2, 1
            elif rate > 0: rubric["logic"], rubric["edge_case"] = 1, 0
            else: rubric["logic"], rubric["edge_case"] = 0, 0
        return rubric

    def sanitize_rubric(self, rubric: dict, fallback: dict) -> dict:
        clean = {}
        for dim, (lo, hi) in self.RUBRIC_RANGES.items():
            val = rubric.get(dim, fallback.get(dim, 0))
            try: val = int(float(val))
            except: val = fallback.get(dim, 0)
            clean[dim] = max(lo, min(hi, val))
        return clean

    def normalize_taxonomy(self, labels):
        if labels is None: return []
        clean = []
        for label in labels:
            if not isinstance(label, str): continue
            label = label.strip()
            mapped = label if label in self.OFFICIAL_LABELS else self.LABEL_MAP.get(label)
            if mapped and mapped not in clean:
                clean.append(mapped)
        return clean

    def _parse_response(self, response: str, compile_log: str = "", 
                        test_report=None, code: str = "") -> tuple:
        """Parse JSON từ model output với fallback thông minh."""
        import re
        
        # Bỏ markdown code blocks
        clean_response = response.strip()
        clean_response = re.sub(r'^```json\s*', '', clean_response)
        clean_response = re.sub(r'^```\s*', '', clean_response)
        clean_response = re.sub(r'\s*```$', '', clean_response)
        
        # Tìm JSON object
        json_match = re.search(r'\{[^{}]*"rubric"[^{}]*\{[^}]*\}[^}]*"errors"[^}]*\[[^\]]*\][^}]*\}', 
                            clean_response, re.DOTALL)
        
        if not json_match:
            # Thử tìm bất kỳ JSON nào
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
        
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                rubric = parsed.get("rubric", {})
                errors = parsed.get("errors", [])
                
                # Validate rubric
                valid_rubric = {}
                for dim in self.RUBRIC_DIMS:
                    val = rubric.get(dim, 0)
                    try:
                        val = int(float(val))
                    except:
                        val = 0
                    lo, hi = self.RUBRIC_RANGES[dim]
                    valid_rubric[dim] = max(lo, min(hi, val))
                
                # Validate errors
                valid_errors = []
                if isinstance(errors, list):
                    for e in errors:
                        if isinstance(e, str) and e in self.OFFICIAL_LABELS:
                            valid_errors.append(e)
                
                # Compiler-based correction
                has_compile_error = "error" in (compile_log or "").lower()
                if has_compile_error:
                    valid_rubric["compilable"] = 0
                    valid_rubric["logic"] = 0
                    valid_rubric["edge_case"] = 0
                elif valid_rubric["compilable"] == 0:
                    # Nếu model nói compilable=0 nhưng log không có error → sửa lại
                    if compile_log and "error" not in compile_log.lower():
                        valid_rubric["compilable"] = 1
                
                return valid_rubric, valid_errors, response
                
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        
        # Fallback: heuristic dựa trên compile_log
        fallback_rubric = self._heuristic_rubric(compile_log, test_report, code)
        fallback_errors = self._heuristic_errors(compile_log, test_report, code)
        return fallback_rubric, fallback_errors, response

    def _heuristic_rubric(self, compile_log: str, test_report, code: str) -> dict:
        """Fallback rubric khi parse fail."""
        has_error = "error" in (compile_log or "").lower()
        
        if has_error:
            return {d: 0 for d in self.RUBRIC_DIMS}
        
        rubric = {
            "compilable": 1, "io_format": 1, "logic": 2,
            "edge_case": 1, "complexity": 1, "code_quality": 1
        }
        
        # Điều chỉnh dựa trên test_report
        if test_report and isinstance(test_report, list) and len(test_report) > 0:
            passed = sum(1 for t in test_report if isinstance(t, dict) and t.get("passed", False))
            total = len(test_report)
            rate = passed / total if total > 0 else 0
            
            if rate == 1.0:
                rubric["logic"] = 4
                rubric["edge_case"] = 2
            elif rate >= 0.75:
                rubric["logic"] = 3
                rubric["edge_case"] = 1
            elif rate >= 0.5:
                rubric["logic"] = 2
            elif rate > 0:
                rubric["logic"] = 1
            else:
                rubric["logic"] = 0
        
        # Điều chỉnh dựa trên code quality
        code_len = len(code.strip()) if code else 0
        if code_len < 50:
            rubric["code_quality"] = 0
        elif code_len < 200:
            rubric["code_quality"] = 1
        
        return rubric


    def _heuristic_errors(self, compile_log: str, test_report, code: str) -> list:
        """Fallback errors khi parse fail."""
        errors = []
        log_lower = (compile_log or "").lower()
        
        if "error" in log_lower:
            errors.append("Lỗi biên dịch")
            if "expected" in log_lower or "syntax" in log_lower:
                errors.append("Lỗi cú pháp")
            if "no matching function" in log_lower or "no match" in log_lower:
                errors.append("Lỗi hàm")
            if "cannot convert" in log_lower or "invalid conversion" in log_lower:
                errors.append("Lỗi kiểu dữ liệu")
        
        if test_report and isinstance(test_report, list):
            failed = [t for t in test_report if isinstance(t, dict) and not t.get("passed", True)]
            if failed:
                errors.append("Lỗi logic")
        
        code_lower = (code or "").lower()
        if "while(true)" in code_lower or "while(1)" in code_lower:
            errors.append("Lỗi vòng lặp")
        
        return list(set(errors))

    def grade_batch(self, submissions: List[dict]) -> List[GradingResult]:
        results = []
        for i, sub in enumerate(submissions):
            print(f"  Grading [{i+1}/{len(submissions)}] {sub.get('sample_id', 'unknown')}...")
            result = self.grade(
                statement=sub.get("statement", ""), code=sub.get("code", ""),
                compile_log=sub.get("compile_log", ""), test_report=sub.get("test_report", ""),
            )
            result.sample_id = sub.get("sample_id", "")
            results.append(result)
        return results

    def apply_prerequisite_policy(self, results: List[GradingResult], 
                            submissions: List[dict]) -> List[GradingResult]:
        """
        Áp dụng chính sách tiên quyết CHỈ cho đề multi_problem.
        Chỉ ép về 0 nếu câu TRƯỚC ĐÓ compilable=0.
        KHÔNG ép về 0 nếu chỉ logic=0.
        """
        exam_groups = {}
        for i, result in enumerate(results):
            exam_id = submissions[i].get("exam_id", "")
            exam_type = submissions[i].get("exam_type", "")
            if exam_id not in exam_groups:
                exam_groups[exam_id] = {"type": exam_type, "items": []}
            exam_groups[exam_id]["items"].append((i, result))
        
        for exam_id, group_data in exam_groups.items():
            # CHỈ áp cho multi_problem
            if group_data["type"] != "multi_problem":
                continue
                
            group = group_data["items"]
            group.sort(key=lambda x: x[1].sample_id)
            
            has_failed_prerequisite = False
            for idx, result in group:
                # ép về 0 nếu không compile được
                if has_failed_prerequisite:
                    result.rubric = {d: 0 for d in self.RUBRIC_DIMS}
                elif result.rubric.get("compilable", 0) == 0:
                    result.rubric = {d: 0 for d in self.RUBRIC_DIMS}
                    has_failed_prerequisite = True
        
        return results