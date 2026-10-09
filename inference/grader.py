"""
inference/grader.py
Task 1 + Task 2: Chấm điểm rubric + phân loại lỗi
Sử dụng transformers + LoRA adapter, greedy decoding cho resilience.
"""
import json
import torch
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Cố định seed cho resilience
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
    
    ERROR_LABELS = [
        "Lỗi biên dịch", "Lỗi logic", "Lỗi vòng lặp", "Lỗi hàm",
        "Lỗi I/O", "Lỗi bộ nhớ", "Lỗi kiểu dữ liệu", "Lỗi thuật toán",
        "Lỗi xử lý biên", "Không có lỗi"
    ]

    def __init__(self, base_model: str, adapter_path: str = None, device: str = "cuda"):
        self.device = device
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load base model + LoRA adapter."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        print(f"Loading base model: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True,
        )

        # Load LoRA adapter nếu có
        if self.adapter_path and os.path.exists(self.adapter_path):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload()
            print(f"✅ Loaded LoRA adapter from {self.adapter_path}")

        self.model.eval()

        # Generation config: greedy decoding cho resilience
        self.gen_config = GenerationConfig(
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def build_prompt(self, statement: str, code: str, compile_log: str,
                     test_report: str = "") -> str:
        """Xây dựng prompt ChatML format."""
        compile_status = "COMPILE SUCCESS" if "error" not in compile_log.lower() else "COMPILE FAILED"

        input_text = (
            f"### Đề bài:\n{statement}\n\n"
            f"### Code sinh viên:\n```cpp\n{code}\n```\n\n"
            f"### Kết quả biên dịch thực tế (g++ -std=c++17):\n{compile_status}\n"
            f"Log: {compile_log[:500] if compile_log else 'Empty'}\n\n"
            f"### Test Report:\n{test_report if test_report else 'Không có test case công khai'}"
        )

        prompt = (
            f"<|im_start|>system\n"
            f"You are an expert C++ code grader. You have ACTUAL COMPILATION RESULTS as ground truth. "
            f"Use them to grade accurately. Output JSON with rubric scores and error labels.<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def grade(self, statement: str, code: str, compile_log: str,
              test_report: str = "") -> GradingResult:
        """Chấm điểm một bài nộp."""
        prompt = self.build_prompt(statement, code, compile_log, test_report)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=self.gen_config)

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Parse JSON từ response
        rubric, errors, rationale = self._parse_response(response)

        # Post-processing: Rule-based correction dựa trên compiler
        rubric = self._apply_compiler_rules(rubric, compile_log)

        return GradingResult(
            sample_id="",
            rubric=rubric,
            errors=errors,
            rationale=rationale,
            compiler_verified=True,
        )

    def _apply_compiler_rules(self, rubric: dict, compile_log: str) -> dict:
        """Áp dụng luật cứng dựa trên kết quả compiler."""
        has_error = "error" in compile_log.lower()

        if has_error:
            rubric["compilable"] = 0
            rubric["logic"] = 0
            rubric["edge_case"] = 0
            rubric["complexity"] = 0
        else:
            rubric["compilable"] = 1

        # Clamp values
        rubric["compilable"] = max(0, min(1, rubric.get("compilable", 0)))
        rubric["io_format"] = max(0, min(1, rubric.get("io_format", 0)))
        rubric["logic"] = max(0, min(4, rubric.get("logic", 0)))
        rubric["edge_case"] = max(0, min(2, rubric.get("edge_case", 0)))
        rubric["complexity"] = max(0, min(1, rubric.get("complexity", 0)))
        rubric["code_quality"] = max(0, min(1, rubric.get("code_quality", 0)))

        return rubric

    def _parse_response(self, response: str) -> tuple:
        """Parse JSON từ model output."""
        import re

        # Tìm JSON block
        json_match = re.search(
            r'\{[^{}]*"rubric"[^{}]*\{[^}]*\}[^}]*"errors"[^}]*\[[^\]]*\][^}]*\}',
            response, re.DOTALL
        )

        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("rubric", {}), data.get("errors", []), response
            except json.JSONDecodeError:
                pass

        # Fallback: trả về rubric mặc định
        default_rubric = {
            "compilable": 0, "io_format": 0, "logic": 0,
            "edge_case": 0, "complexity": 0, "code_quality": 0
        }
        return default_rubric, [], response

    def grade_batch(self, submissions: List[dict]) -> List[GradingResult]:
        """Chấm điểm hàng loạt."""
        results = []
        for i, sub in enumerate(submissions):
            print(f"  Grading [{i+1}/{len(submissions)}] {sub.get('sample_id', 'unknown')}...")
            result = self.grade(
                statement=sub.get("statement", ""),
                code=sub.get("code", ""),
                compile_log=sub.get("compile_log", ""),
                test_report=sub.get("test_report", ""),
            )
            result.sample_id = sub.get("sample_id", "")
            results.append(result)
        return results