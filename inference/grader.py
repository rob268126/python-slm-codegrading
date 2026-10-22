"""
inference/grader.py
Chấm điểm Rubric 6 chiều (Task 1) và Phân loại lỗi Taxonomy (Task 2).

Sử dụng Pydantic v2 để chuẩn hóa schema và loại bỏ toàn bộ config rule thủ công:
1. Đảm bảo 100% nhãn taxonomy trả về thuộc đúng 10 nhãn chính thức của BTC qua Pydantic field validator.
2. Tự động áp dụng miền giá trị (ge, le) và prerequisite policy qua Pydantic model validator.
3. Tích hợp Compiler Sandbox g++ làm ground truth thực tế.
4. Parsing an toàn và tự phục hồi từ LLM response sang Pydantic model `LLMJudgeResponse`.
"""
from __future__ import annotations

import json
import re
from typing import List, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.data import Submission, has_compile_error
from core.schema import (
    LABEL_COMPILE,
    LABEL_HARDCODE,
    LABEL_LOGIC,
    LABEL_LOOP,
    LABEL_STYLE,
    GradingResult,
    Rubric,
    normalize_labels,
)
from sandbox.cpp_compiler import compile_cpp


class LLMJudgeResponse(BaseModel):
    """Pydantic model để parse và tự sửa lỗi từ JSON sinh ra bởi LLM."""
    model_config = ConfigDict(extra="ignore")

    rubric: Rubric = Field(default_factory=Rubric)
    taxonomy_error: List[str] = Field(default_factory=list)

    @field_validator("taxonomy_error", mode="before")
    @classmethod
    def _coerce_taxonomy(cls, v):
        return normalize_labels(v)

    @model_validator(mode="before")
    @classmethod
    def _pre_extract(cls, data):
        # Hỗ trợ cả key chuẩn 'taxonomy_error' lẫn key cũ 'errors'
        if isinstance(data, dict):
            if "taxonomy_error" not in data and "errors" in data:
                data["taxonomy_error"] = data["errors"]
        return data


class CppGrader:
    """Module chấm điểm C++ sử dụng LLM Fine-tuned (Qwen2.5-Coder LoRA) kết hợp Compiler Sandbox."""

    def __init__(self, llm=None, use_compiler_sandbox: bool = True):
        """
        Args:
            llm: Instance của inference.llm.LLM (hoặc None để chạy heuristic baseline).
            use_compiler_sandbox: Nếu True và submission chưa có compile_log, sẽ gọi g++ biên dịch thực tế.
        """
        self.llm = llm
        self.use_compiler_sandbox = use_compiler_sandbox

    def build_prompt(self, statement: str, code: str, compile_log: str, test_report: str = "") -> str:
        """Xây dựng prompt ChatML format khớp với định dạng lúc huấn luyện LoRA."""
        has_error = has_compile_error(compile_log)
        compile_status = "COMPILE FAILED" if has_error else "COMPILE SUCCESS"
        compile_log_display = compile_log[:500] if compile_log else "Không có lỗi biên dịch"

        input_text = (
            f"### Đề bài:\n{statement}\n\n"
            f"### Code sinh viên:\n```cpp\n{code}\n```\n\n"
            f"### Kết quả biên dịch thực tế (g++ -std=c++11):\n{compile_status}\n"
            f"Log: {compile_log_display}\n\n"
            f"### Test Report:\n{test_report if test_report else 'Không có'}"
        )

        prompt = (
            f"<|im_start|>system\n"
            f"You are an expert C++ code grader acting as a JUDGE (Local JEV Engine: Score + Choice).\n"
            f"You have ACTUAL COMPILATION RESULTS as ground truth - trust them above all.\n"
            f"Grading rubric (Score Dimensions):\n"
            f"- compilable (0-1): Does code compile? Use compiler result.\n"
            f"- io_format (0-1): Correct input/output format?\n"
            f"- logic (0-4): Algorithm correctness. IF compilable=0, this MUST be 0.\n"
            f"- edge_case (0-2): Handles edge cases? IF compilable=0, this MUST be 0.\n"
            f"- complexity (0-1): Appropriate time/space complexity?\n"
            f"- code_quality (0-1): Clean, readable code?\n\n"
            f"Official Error Taxonomy (Choice Options - multi-label):\n"
            f"- Lỗi biên dịch: Syntax, type, or linking error preventing compilation.\n"
            f"- Lỗi nhập/xuất: Wrong I/O format, missing or extra output characters.\n"
            f"- Lỗi logic: Incorrect logic or calculation yielding wrong answers.\n"
            f"- Lỗi vòng lặp: Infinite loop, off-by-one loop conditions.\n"
            f"- Lỗi mảng/chuỗi: Out-of-bounds array or string access, index error.\n"
            f"- Lỗi hàm: Missing function implementation, recursion overflow, bad return.\n"
            f"- Lỗi edge case: Failure to handle corner cases (0, empty, overflow).\n"
            f"- Lỗi thuật toán: Inefficient or inappropriate algorithm causing TLE.\n"
            f"- Lỗi hard-code: Hard-coding specific test values.\n"
            f"- Lỗi style: Extremely poor naming, indentation, or dead code.\n\n"
            f"Output ONLY valid JSON: {{\"rubric\": {{\"compilable\": 0, \"io_format\": 0, \"logic\": 0, \"edge_case\": 0, \"complexity\": 0, \"code_quality\": 0}}, \"taxonomy_error\": [...]}}<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"Rationale: \n"
        )
        return prompt

    def _ensure_compile_info(self, sub: Submission) -> Submission:
        """Nếu submission chưa có kết quả compile và sandbox được bật, chạy g++ để lấy ground truth."""
        if sub.compile_failed is not None or not self.use_compiler_sandbox or not sub.code:
            return sub

        res = compile_cpp(sub.code, exam_type=sub.exam_type)
        sub.compile_log = res.get("compile_log", "")
        sub.compile_failed = not res.get("compile_success", False)
        return sub

    def heuristic_grade(self, sub: Submission) -> GradingResult:
        """Heuristic chấm điểm dựa trên compiler sandbox, test report và phân tích mã nguồn."""
        sub = self._ensure_compile_info(sub)
        code = sub.code or ""
        code_lower = code.lower()

        # Kiểm tra compile
        is_failed = sub.compile_failed is True or has_compile_error(sub.compile_log)

        if is_failed or len(code.strip()) == 0:
            return GradingResult(
                sample_id=sub.sample_id,
                rubric={"compilable": 0, "code_quality": 0 if len(code.strip()) < 30 else 1},
                taxonomy_error=[LABEL_COMPILE],
                rationale="Code không biên dịch được hoặc rỗng theo kết quả compiler.",
                compiler_verified=True,
            )

        # Code biên dịch được: khởi tạo điểm mặc định khá
        rubric_dict = {
            "compilable": 1, "io_format": 1, "logic": 2,
            "edge_case": 1, "complexity": 1, "code_quality": 1
        }
        errors: List[str] = []

        # Điều chỉnh theo test_report (nếu có ở single_problem)
        if isinstance(sub.test_report, list) and len(sub.test_report) > 0:
            passed = sum(1 for t in sub.test_report if isinstance(t, dict) and t.get("passed", False))
            total = len(sub.test_report)
            rate = passed / total if total > 0 else 0.0

            if rate == 1.0:
                rubric_dict["logic"], rubric_dict["edge_case"] = 4, 2
            elif rate >= 0.75:
                rubric_dict["logic"], rubric_dict["edge_case"] = 3, 1
            elif rate >= 0.5:
                rubric_dict["logic"], rubric_dict["edge_case"] = 2, 1
            elif rate > 0:
                rubric_dict["logic"], rubric_dict["edge_case"] = 1, 0
                errors.append(LABEL_LOGIC)
            else:
                rubric_dict["logic"], rubric_dict["edge_case"] = 0, 0
                errors.append(LABEL_LOGIC)

        # Heuristic phát hiện lỗi logic/vòng lặp/hardcode/style
        if "while(true)" in code_lower or "while(1)" in code_lower:
            errors.append(LABEL_LOOP)

        # Phát hiện bài nộp bị comment toàn bộ (Lỗi style)
        non_empty_lines = [l.strip() for l in code.splitlines() if l.strip()]
        comment_lines = sum(1 for l in non_empty_lines if l.startswith("//") or l.startswith("/*"))
        if non_empty_lines and (comment_lines / len(non_empty_lines)) > 0.6:
            errors.append(LABEL_STYLE)

        # Phát hiện hardcode tên file csv trong đề multi_problem
        if sub.is_multi_problem and re.search(r'"[^"\n]*\.csv"', code, re.IGNORECASE):
            errors.append(LABEL_HARDCODE)

        return GradingResult(
            sample_id=sub.sample_id,
            rubric=rubric_dict,
            taxonomy_error=errors,
            rationale="Heuristic grading based on compiler log and code analysis.",
            compiler_verified=True,
        )

    def _parse_llm_response(self, response: str, sub: Submission) -> GradingResult:
        """Parse JSON từ response của model và chuẩn hóa qua Pydantic."""
        clean = response.strip()
        # Bỏ markdown code block nếu có
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)

        # Tìm JSON object chứa rubric hoặc errors
        match = re.search(r"\{[^{}]*\"rubric\"[^{}]*\{[^}]*\}[^{}]*\}", clean, re.DOTALL)
        if not match:
            match = re.search(r"\{.*\}", clean, re.DOTALL)

        if not match:
            return self.heuristic_grade(sub)

        try:
            parsed_json = json.loads(match.group())
            judge_res = LLMJudgeResponse.model_validate(parsed_json)
        except Exception:
            # Fallback an toàn sang heuristic nếu JSON hoàn toàn không parse được
            return self.heuristic_grade(sub)

        # Áp dụng ground truth compiler thực tế
        sub = self._ensure_compile_info(sub)
        is_compile_failed = sub.compile_failed is True or has_compile_error(sub.compile_log)

        res = GradingResult(
            sample_id=sub.sample_id,
            rubric=judge_res.rubric,
            taxonomy_error=judge_res.taxonomy_error,
            rationale=clean[:300],
        )
        res.apply_compiler_ground_truth(is_compile_failed)
        return res

    def grade_one(self, sub: Submission) -> GradingResult:
        """Chấm điểm 1 bài nộp."""
        if self.llm is None:
            return self.heuristic_grade(sub)

        sub = self._ensure_compile_info(sub)

        def _render(stmt: str, code: str) -> str:
            return self.build_prompt(
                statement=stmt,
                code=code,
                compile_log=sub.compile_log,
                test_report=json.dumps(sub.test_report) if sub.test_report else "",
            )

        prompt = self.llm.fit_prompt(_render, sub.statement, sub.code)
        response = self.llm.generate(prompt, max_new_tokens=400)
        return self._parse_llm_response(response, sub)

    def grade_batch(self, submissions: Sequence[Submission]) -> List[GradingResult]:
        """Chấm điểm hàng loạt danh sách submission."""
        results: List[GradingResult] = []
        n = len(submissions)
        for i, sub in enumerate(submissions):
            if (i + 1) % 10 == 0 or i == 0 or i == n - 1:
                print(f"  [Grader] [{i+1}/{n}] Grading {sub.sample_id} ({sub.exam_id})...")
            res = self.grade_one(sub)
            results.append(res)
        return results