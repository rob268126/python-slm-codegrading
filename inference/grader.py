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
    LABEL_IO,
    LABEL_LOGIC,
    LABEL_LOOP,
    LABEL_STYLE,
    RUBRIC_DIMS,
    GradingResult,
    Rubric,
    normalize_labels,
)
from sandbox.cpp_compiler import compile_cpp
from sandbox.differential import filter_exam_reports, summarize_report


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


def apply_test_evidence(rubric, errors: List[str], report) -> List[str]:
    """Ghi đè các chiều rubric phụ thuộc hành vi chạy bằng pass-rate của test thực thi. Trả về taxonomy đã chỉnh.

    logic/edge_case dùng pass-rate "lỏng" (đúng đáp án, bỏ qua lời nhắc/hoa-thường) để lỗi định dạng
    không kéo theo mất điểm logic; io_format chỉ bị trừ khi pass-rate chính xác thấp hơn pass-rate lỏng.
    """
    stats = summarize_report(report)
    if stats is None or not getattr(rubric, "compilable", 1):
        return errors
    rate, exact_rate, timeouts = stats["lenient_rate"], stats["rate"], stats["timeouts"]
    errors = list(errors)

    if rate == 1.0:
        rubric.logic = 4
        rubric.edge_case = max(rubric.edge_case, 1)
        rubric.complexity = 1
        errors = [e for e in errors if e != LABEL_LOGIC]
    elif rate >= 0.75:
        rubric.logic = 3
        rubric.edge_case = min(rubric.edge_case, 1)
        errors.append(LABEL_LOGIC)
    elif rate >= 0.5:
        rubric.logic = 2
        rubric.edge_case = min(rubric.edge_case, 1)
        errors.append(LABEL_LOGIC)
    else:
        rubric.logic = 1 if rate > 0 else 0
        rubric.edge_case = 0
        errors.append(LABEL_LOGIC)

    if rate == 0:
        rubric.io_format = 0
    elif exact_rate < rate:
        rubric.io_format = 0
        errors.append(LABEL_IO)
    elif exact_rate == 1.0:
        rubric.io_format = 1
        errors = [e for e in errors if e != LABEL_IO]

    if timeouts:
        # Input sinh tự động đều nhỏ (< 300 ký tự) -> timeout gần như chắc chắn là vòng lặp vô hạn.
        rubric.complexity = 0
        errors.append(LABEL_LOOP)
    return errors


class CppGrader:
    """Module chấm điểm C++ sử dụng LLM Fine-tuned (Qwen2.5-Coder LoRA) kết hợp Compiler Sandbox."""

    def __init__(self, llm=None, use_compiler_sandbox: bool = True, tester=None, report_in_prompt: bool = False):
        """
        Args:
            llm: Instance của inference.llm.LLM (hoặc None để chạy heuristic baseline).
            use_compiler_sandbox: Nếu True và submission chưa có compile_log, sẽ gọi g++ biên dịch thực tế.
            tester: DifferentialTester sinh test_report khi dataset không cung cấp (chỉ đề single_problem).
            report_in_prompt: Đưa test_report vào prompt LLM (chỉ bật khi adapter được train với test_report).
        """
        self.llm = llm
        self.use_compiler_sandbox = use_compiler_sandbox
        self.tester = tester
        self.report_in_prompt = report_in_prompt
        self._reported: set = set()

    def _ensure_test_report(self, sub: Submission) -> Submission:
        if (sub.test_report or self.tester is None or sub.is_multi_problem or sub.compile_failed
                or sub.sample_id in self._reported):
            return sub
        sub.test_report = self.tester.build_report(sub.exam_id, sub.statement, sub.code)
        self._reported.add(sub.sample_id)
        return sub

    def prepare_test_reports(self, submissions: Sequence[Submission]) -> None:
        """Compile + chạy test cho cả lô trước, rồi kiểm tra chéo bộ test theo từng đề (bỏ test sai)."""
        if self.tester is None:
            return
        by_exam: dict = {}
        for sub in submissions:
            if sub.is_multi_problem or sub.test_report:
                continue
            self._ensure_compile_info(sub)
            self._ensure_test_report(sub)
            by_exam.setdefault(sub.exam_id, []).append(sub)
        for exam_id, subs in by_exam.items():
            filtered = filter_exam_reports([s.test_report for s in subs], exam_id=exam_id)
            for s, rep in zip(subs, filtered):
                s.test_report = rep

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
            f"You are an expert C++ code grader acting as a JUDGE.\n"
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
            f"- Lỗi hard-code: Hard-coding specific test values or file names.\n"
            f"- Lỗi style: Extremely poor naming, indentation, or dead code.\n\n"
            f"Output ONLY valid JSON with keys \"rubric\" and \"taxonomy_error\":\n"
            f"{{\"rubric\": {{\"compilable\": 1, \"io_format\": 1, \"logic\": 2, \"edge_case\": 1, \"complexity\": 1, \"code_quality\": 1}}, \"taxonomy_error\": [...]}}<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def _ensure_compile_info(self, sub: Submission) -> Submission:
        """Nếu submission chưa có kết quả compile và sandbox được bật, chạy g++ để lấy ground truth."""
        if sub.compile_failed is not None or not self.use_compiler_sandbox or not sub.code:
            return sub

        res = compile_cpp(sub.code, exam_type=sub.exam_type)
        success = res.get("compile_success")
        if success is None:
            # Không có g++ / timeout: giữ trạng thái "chưa biết", không phạt bài làm.
            return sub
        sub.compile_log = res.get("compile_log", "")
        sub.compile_failed = not success
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

        # Code biên dịch được: khởi tạo điểm mặc định chuẩn theo dữ liệu phân bố của giảng viên (median = 7)
        rubric_dict = {
            "compilable": 1, "io_format": 1, "logic": 2,
            "edge_case": 1, "complexity": 1, "code_quality": 1
        }
        errors: List[str] = []

        # Heuristic phát hiện lỗi logic / vòng lặp
        if "while(true)" in code_lower or "while(1)" in code_lower:
            errors.append(LABEL_LOOP)

        # Phát hiện bài nộp bị comment toàn bộ hoặc tỷ lệ comment bất thường (Lỗi style)
        non_empty_lines = [l.strip() for l in code.splitlines() if l.strip()]
        comment_lines = sum(1 for l in non_empty_lines if l.startswith("//") or l.startswith("/*"))
        if non_empty_lines and (comment_lines / len(non_empty_lines)) > 0.45:
            errors.append(LABEL_STYLE)
            rubric_dict["code_quality"] = 0

        # Hard-code tên file csv trong các hàm của đề multi_problem (main tự viết để test thì được phép)
        if sub.is_multi_problem:
            body = re.split(r"\bint\s+main\s*\(", code, maxsplit=1)[0]
            if re.search(r'"[^"\n]*\.csv"', body, re.IGNORECASE) or re.search(r'#\s*include\s*"[^"\n]*\.csv"', code, re.IGNORECASE):
                errors.append(LABEL_HARDCODE)

        res = GradingResult(
            sample_id=sub.sample_id,
            rubric=rubric_dict,
            taxonomy_error=errors,
            rationale="Heuristic grading based on compiler log and code analysis.",
            compiler_verified=True,
        )
        if sub.test_report:
            res.taxonomy_error = normalize_labels(apply_test_evidence(res.rubric, res.taxonomy_error, sub.test_report))
            res.total_score = res.rubric.total_score
        return res

    @staticmethod
    def _salvage_json(text: str):
        """Cứu JSON hỏng/bị cắt: lấy từng chiều rubric và các nhãn hợp lệ bằng regex."""
        rubric = {}
        for dim in RUBRIC_DIMS:
            m = re.search(rf'"{dim}"\s*:\s*(\d+)', text)
            if m:
                rubric[dim] = int(m.group(1))
        if not rubric:
            return None
        labels = normalize_labels(re.findall(r'"([^"\n]+)"', text.split('"taxonomy_error"', 1)[-1])) \
            if '"taxonomy_error"' in text else []
        return {"rubric": rubric, "taxonomy_error": labels}

    def _parse_llm_response(self, response: str, sub: Submission) -> GradingResult:
        """Parse JSON từ response của model và chuẩn hóa qua Pydantic."""
        clean = response.strip()
        # Bỏ markdown code block nếu có
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
        if not clean.startswith("{"):
            clean = "{" + clean

        parsed_json = None
        match = re.search(r"\{[^{}]*\"rubric\"[^{}]*\{[^}]*\}[^{}]*\}", clean, re.DOTALL) \
            or re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                parsed_json = json.loads(match.group())
            except Exception:
                parsed_json = None
        if not isinstance(parsed_json, dict) or not isinstance(parsed_json.get("rubric"), dict):
            parsed_json = self._salvage_json(clean)
        if parsed_json is None:
            return self.heuristic_grade(sub)

        # Áp dụng ground truth compiler thực tế
        sub = self._ensure_compile_info(sub)
        is_compile_failed = sub.compile_failed is True or has_compile_error(sub.compile_log)
        heuristic_res = self.heuristic_grade(sub)

        # Chiều nào LLM bỏ sót thì lấy theo heuristic (mặc định trung vị), không để Pydantic điền 0.
        # Đồng bộ compilable TRƯỚC KHI validate để prerequisite policy không zero-out oan.
        raw_rubric = {**heuristic_res.rubric.to_dict(), **parsed_json["rubric"]}
        raw_rubric["compilable"] = 0 if is_compile_failed else 1
        parsed_json["rubric"] = raw_rubric

        try:
            judge_res = LLMJudgeResponse.model_validate(parsed_json)
        except Exception:
            return heuristic_res

        # Bổ sung các nhãn lỗi code-level khách quan (hard-code, style) nếu phát hiện trong code
        combined_errors = set(judge_res.taxonomy_error)
        for err in heuristic_res.taxonomy_error:
            if err in (LABEL_HARDCODE, LABEL_STYLE):
                combined_errors.add(err)

        res = GradingResult(
            sample_id=sub.sample_id,
            rubric=judge_res.rubric,
            taxonomy_error=list(combined_errors),
            rationale=clean[:300],
        )
        res.apply_compiler_ground_truth(is_compile_failed)
        if not is_compile_failed and sub.test_report:
            res.taxonomy_error = normalize_labels(
                apply_test_evidence(res.rubric, res.taxonomy_error, sub.test_report))
            res.total_score = res.rubric.total_score
        return res

    def grade_one(self, sub: Submission) -> GradingResult:
        """Chấm điểm 1 bài nộp."""
        sub = self._ensure_compile_info(sub)
        sub = self._ensure_test_report(sub)

        if self.llm is None:
            return self.heuristic_grade(sub)

        report_text = json.dumps(sub.test_report, ensure_ascii=False) if (sub.test_report and self.report_in_prompt) else ""

        def _render(stmt: str, code: str) -> str:
            return self.build_prompt(
                statement=stmt,
                code=code,
                compile_log=sub.compile_log,
                test_report=report_text,
            )

        prompt = self.llm.fit_prompt(_render, sub.statement, sub.code)
        response = self.llm.generate(prompt, max_new_tokens=400)
        return self._parse_llm_response(response, sub)

    def grade_batch(self, submissions: Sequence[Submission]) -> List[GradingResult]:
        """Chấm điểm hàng loạt danh sách submission."""
        results: List[GradingResult] = []
        n = len(submissions)
        self.prepare_test_reports(submissions)
        for i, sub in enumerate(submissions):
            if (i + 1) % 10 == 0 or i == 0 or i == n - 1:
                print(f"  [Grader] [{i+1}/{n}] Grading {sub.sample_id} ({sub.exam_id})...")
            res = self.grade_one(sub)
            results.append(res)
        return results