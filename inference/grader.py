"""
inference/grader.py
Chấm điểm Rubric 6 chiều (Task 1) và Phân loại lỗi Taxonomy (Task 2).

Các điểm cải tiến cốt lõi:
1. Đảm bảo 100% nhãn taxonomy trả về thuộc đúng 10 nhãn chính thức của BTC qua `normalize_labels`.
   Khắc phục dứt điểm lỗi bị Leaderboard từ chối: 'giá trị ở "/taxonomy_error/..." chưa đúng dạng'.
2. Tích hợp Compiler Sandbox g++ làm ground truth:
   - Khi bài không biên dịch được -> compilable=0, logic=0, edge_case=0, complexity=0, gắn nhãn 'Lỗi biên dịch'.
3. Heuristic fallback đa tầng: Nếu LLM parse JSON lỗi, fallback dựa trên compile log, test report và phân tích code.
4. Loại bỏ hoàn toàn bug 'prerequisite policy' cũ (vốn làm 0 điểm hàng loạt sinh viên khác nhau).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from core.data import Submission, has_compile_error
from core.schema import (
    LABEL_COMPILE,
    LABEL_HARDCODE,
    LABEL_IO,
    LABEL_LOGIC,
    LABEL_LOOP,
    LABEL_STYLE,
    RUBRIC_DIMS,
    clamp_rubric,
    normalize_labels,
    total_score,
)
from sandbox.cpp_compiler import compile_cpp


@dataclass
class GradingResult:
    sample_id: str
    rubric: Dict[str, int]
    taxonomy_error: List[str]
    total_score: int = 0
    rationale: str = ""
    compiler_verified: bool = False

    def __post_init__(self):
        self.rubric = clamp_rubric(self.rubric)
        self.taxonomy_error = normalize_labels(self.taxonomy_error)
        self.total_score = total_score(self.rubric)


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
            rubric = {
                "compilable": 0, "io_format": 0, "logic": 0,
                "edge_case": 0, "complexity": 0, "code_quality": 0 if len(code.strip()) < 30 else 1
            }
            errors = [LABEL_COMPILE]
            return GradingResult(
                sample_id=sub.sample_id,
                rubric=rubric,
                taxonomy_error=errors,
                rationale="Code không biên dịch được hoặc rỗng theo kết quả compiler.",
                compiler_verified=True,
            )

        # Code biên dịch được: khởi tạo điểm mặc định khá
        rubric = {
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
                rubric["logic"], rubric["edge_case"] = 4, 2
            elif rate >= 0.75:
                rubric["logic"], rubric["edge_case"] = 3, 1
            elif rate >= 0.5:
                rubric["logic"], rubric["edge_case"] = 2, 1
            elif rate > 0:
                rubric["logic"], rubric["edge_case"] = 1, 0
                errors.append(LABEL_LOGIC)
            else:
                rubric["logic"], rubric["edge_case"] = 0, 0
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
            rubric=rubric,
            taxonomy_error=errors,
            rationale="Heuristic grading based on compiler log and code analysis.",
            compiler_verified=True,
        )

    def _parse_llm_response(self, response: str, sub: Submission) -> GradingResult:
        """Parse JSON từ response của model và chuẩn hóa nhãn."""
        clean = response.strip()
        # Bỏ markdown block nếu có
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)

        parsed_json = None
        # Tìm JSON object chứa rubric hoặc errors
        match = re.search(r"\{[^{}]*\"rubric\"[^{}]*\{[^}]*\}[^{}]*\}", clean, re.DOTALL)
        if not match:
            match = re.search(r"\{.*\}", clean, re.DOTALL)

        if match:
            try:
                parsed_json = json.loads(match.group())
            except Exception:
                pass

        if not parsed_json or not isinstance(parsed_json, dict):
            # Fallback sang heuristic nếu model không trả về JSON hợp lệ
            return self.heuristic_grade(sub)

        raw_rubric = parsed_json.get("rubric", {})
        raw_errors = parsed_json.get("errors", [])
        if not isinstance(raw_errors, list):
            raw_errors = []

        # Chuẩn hóa rubric & nhãn lỗi theo đúng chuẩn BTC
        rubric = clamp_rubric(raw_rubric)
        taxonomy_error = normalize_labels(raw_errors)

        # Áp dụng quy tắc compiler cứng (ground truth thực tế)
        sub = self._ensure_compile_info(sub)
        is_compile_failed = sub.compile_failed is True or has_compile_error(sub.compile_log)

        if is_compile_failed:
            rubric["compilable"] = 0
            rubric["logic"] = 0
            rubric["edge_case"] = 0
            rubric["complexity"] = 0
            if LABEL_COMPILE not in taxonomy_error:
                taxonomy_error.append(LABEL_COMPILE)
                taxonomy_error = normalize_labels(taxonomy_error)
        else:
            # Nếu compiler báo thành công mà model đoán nhầm compilable=0 -> sửa lại
            if rubric["compilable"] == 0 and not is_compile_failed:
                rubric["compilable"] = 1
            # Bỏ nhãn 'Lỗi biên dịch' nếu code thực tế compile hoàn toàn tốt
            if not is_compile_failed and LABEL_COMPILE in taxonomy_error:
                taxonomy_error = [e for e in taxonomy_error if e != LABEL_COMPILE]

        return GradingResult(
            sample_id=sub.sample_id,
            rubric=rubric,
            taxonomy_error=taxonomy_error,
            rationale=clean[:300],
            compiler_verified=True,
        )

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