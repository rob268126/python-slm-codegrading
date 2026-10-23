"""
sandbox/differential.py
Differential testing cho đề single_problem khi dataset không có test_report.

Quy trình (hoàn toàn tự động, không gán nhãn thủ công):
1. LLM (tắt LoRA) sinh N lời giải tham chiếu + một bộ stdin test cho từng đề.
2. Chạy các lời giải tham chiếu trên bộ stdin; lấy output theo đa số (>= 2 lời giải đồng thuận) làm expected.
3. Chạy code sinh viên trên cùng bộ stdin -> test_report cùng định dạng với test_report của BTC
   ({input, expected, actual, passed}), thêm cờ `timed_out`.
Kết quả theo đề được cache ra đĩa để tái lập và tránh sinh lại.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

from sandbox.cpp_compiler import CompiledProgram

CODE_BLOCK_RE = re.compile(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
MAX_INPUTS = 16
MIN_VALID_INPUTS = 4
FIELD_CHARS = 120


def _chatml(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def extract_cpp(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    if "```" in text:  # block bị cắt do hết token
        return text.split("```", 1)[1].split("\n", 1)[-1].strip()
    return text.strip()


def parse_inputs(text: str) -> List[str]:
    """Lấy mảng JSON các chuỗi stdin từ output của LLM (chịu được rác xung quanh)."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    seen, result = set(), []
    for item in arr:
        if not isinstance(item, str) or not item.strip() or item in seen:
            continue
        seen.add(item)
        result.append(item if item.endswith("\n") else item + "\n")
    return result[:MAX_INPUTS]


def normalize_output(text: str) -> str:
    return " ".join(text.split())


def _num(tok: str) -> Optional[float]:
    try:
        return float(tok)
    except ValueError:
        return None


def _tok_eq(a: str, b: str) -> bool:
    if a.lower() == b.lower():
        return True
    x, y = _num(a), _num(b)
    return x is not None and y is not None and abs(x - y) <= 1e-6 * max(1.0, abs(y))


def lenient_match(actual: str, expected: str, answer_vocab: set) -> bool:
    """Đúng đáp án nhưng sai định dạng: in thêm lời nhắc/nhãn ("Nhap n:", "Ket qua:"), khác hoa/thường, 5 vs 5.0.

    Các token đáp án phải xuất hiện đúng thứ tự trong output; token thừa chỉ được là chữ (không chứa số)
    và không trùng với bất kỳ token đáp án nào của bộ test (tránh in cả "True False").
    """
    exp, act = expected.split(), actual.split()
    if not exp:
        return not act
    i = 0
    for tok in act:
        if i < len(exp) and _tok_eq(tok, exp[i]):
            i += 1
        elif any(ch.isdigit() for ch in tok) or tok.lower() in answer_vocab:
            return False
    return i == len(exp)


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= FIELD_CHARS else text[:FIELD_CHARS] + "..."


class DifferentialTester:
    def __init__(self, llm, cache_dir: str = "./cache/differential", n_refs: int = 3,
                 run_timeout: float = 2.0, log: Callable[[str], None] = print):
        self.llm = llm
        self.cache_dir = Path(cache_dir)
        self.n_refs = n_refs
        self.run_timeout = run_timeout
        self.log = log
        self._tests: Dict[str, Optional[dict]] = {}

    # ---------------- bộ test theo đề ----------------
    def _cache_path(self, exam_id: str, statement: str) -> Path:
        digest = hashlib.sha1(statement.encode("utf-8")).hexdigest()[:8]
        return self.cache_dir / f"{exam_id}-{digest}.json"

    def _gen_inputs(self, statement: str) -> List[str]:
        system = "You are an expert competitive-programming problem setter."
        user = (
            f"{{statement}}\n\n"
            "Write 14 diverse, VALID stdin inputs for this problem that respect its input format and constraints: "
            "include tiny cases, boundary cases (minimum/maximum allowed values, all-equal, sorted, reversed, "
            "negative numbers if allowed, 0 or empty-like cases if allowed) and a few typical cases. "
            "Keep every input short (under 300 characters). "
            "Answer with ONLY a JSON array of strings, each string being the full stdin (use \\n for newlines)."
        )

        def render(stmt: str, _code: str) -> str:
            return _chatml(system, user.replace("{statement}", stmt))

        prompt = self.llm.fit_prompt(render, statement, "")
        return parse_inputs(self.llm.generate(prompt, max_new_tokens=1200, use_adapter=False))

    def _gen_references(self, statement: str) -> List[str]:
        system = "You are an expert C++ programmer."
        user = (
            "{statement}\n\n"
            "Write a correct and efficient complete C++11 program (with main) that reads stdin and writes stdout "
            "exactly as the problem specifies. Output ONLY the code in one ```cpp block."
        )

        def render(stmt: str, _code: str) -> str:
            return _chatml(system, user.replace("{statement}", stmt))

        prompt = self.llm.fit_prompt(render, statement, "")
        refs = []
        for i in range(self.n_refs):
            temperature = 0.0 if i == 0 else 0.5
            text = self.llm.generate(prompt, max_new_tokens=900, temperature=temperature,
                                     use_adapter=False, seed=42 + i)
            refs.append(extract_cpp(text))
        return refs

    def _build_tests(self, exam_id: str, statement: str) -> Optional[dict]:
        inputs = self._gen_inputs(statement)
        if len(inputs) < MIN_VALID_INPUTS:
            self.log(f"  [Diff] {exam_id}: LLM chỉ sinh {len(inputs)} input hợp lệ -> bỏ qua differential testing")
            return None

        outputs_per_ref: List[List[Optional[str]]] = []
        for code in self._gen_references(statement):
            row: List[Optional[str]] = []
            with CompiledProgram(code) as prog:
                for inp in inputs:
                    if not prog.ok:
                        row.append(None)
                        continue
                    r = prog.run(inp, self.run_timeout)
                    ok = not r["timed_out"] and r["returncode"] == 0 and r["output"].strip()
                    row.append(normalize_output(r["output"]) if ok else None)
            if any(v is not None for v in row):
                outputs_per_ref.append(row)

        if not outputs_per_ref:
            self.log(f"  [Diff] {exam_id}: không có lời giải tham chiếu nào chạy được -> bỏ qua")
            return None

        need = 2 if len(outputs_per_ref) >= 2 else 1
        kept_inputs, expected = [], []
        for j, inp in enumerate(inputs):
            votes = Counter(row[j] for row in outputs_per_ref if row[j] is not None)
            if not votes:
                continue
            value, count = votes.most_common(1)[0]
            if count >= need:
                kept_inputs.append(inp)
                expected.append(value)

        if len(kept_inputs) < MIN_VALID_INPUTS:
            self.log(f"  [Diff] {exam_id}: chỉ {len(kept_inputs)} input có đồng thuận -> bỏ qua")
            return None
        self.log(f"  [Diff] {exam_id}: {len(kept_inputs)} test, {len(outputs_per_ref)} lời giải tham chiếu hợp lệ")
        return {"inputs": kept_inputs, "expected": expected, "n_refs_ok": len(outputs_per_ref)}

    def get_tests(self, exam_id: str, statement: str) -> Optional[dict]:
        if exam_id in self._tests:
            return self._tests[exam_id]
        path = self._cache_path(exam_id, statement)
        tests = None
        if path.exists():
            tests = json.loads(path.read_text(encoding="utf-8"))
        else:
            tests = self._build_tests(exam_id, statement)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(tests, ensure_ascii=False, indent=1), encoding="utf-8")
        self._tests[exam_id] = tests
        return tests

    # ---------------- test_report cho từng bài nộp ----------------
    def build_report(self, exam_id: str, statement: str, code: str) -> Optional[List[dict]]:
        """None nếu không có bộ test hoặc code không biên dịch được (khi đó compiler đã đủ tín hiệu)."""
        tests = self.get_tests(exam_id, statement)
        if not tests or not code.strip():
            return None
        report = []
        vocab = {t.lower() for exp in tests["expected"] for t in exp.split()}
        with CompiledProgram(code) as prog:
            if not prog.ok:
                return None
            for inp, exp in zip(tests["inputs"], tests["expected"]):
                r = prog.run(inp, self.run_timeout)
                actual = normalize_output(r["output"])
                exact = (not r["timed_out"]) and actual == exp
                report.append({
                    "input": _clip(inp),
                    "expected": _clip(exp),
                    "actual": "TIMEOUT" if r["timed_out"] else _clip(actual),
                    "passed": exact,
                    "lenient_passed": exact or ((not r["timed_out"]) and lenient_match(actual, exp, vocab)),
                    "timed_out": r["timed_out"],
                })
        return report


def filter_exam_reports(reports: List[Optional[List[dict]]], min_students: int = 5,
                        log: Callable[[str], None] = print, exam_id: str = "") -> List[Optional[List[dict]]]:
    """Kiểm tra chéo bộ test sinh tự động bằng chính bài làm của cả lớp (không cần nhãn).

    - Test mà KHÔNG sinh viên nào qua (kể cả so khớp lỏng) nhiều khả năng sai đề/sai ràng buộc -> bỏ.
    - Nếu còn quá ít test hợp lệ -> bỏ toàn bộ bằng chứng thực thi của đề này (để LLM tự chấm).
    """
    present = [r for r in reports if r]
    if len(present) < min_students:
        return reports
    n_tests = min(len(r) for r in present)
    keep = [j for j in range(n_tests) if any(r[j].get("lenient_passed") for r in present)]
    if len(keep) < MIN_VALID_INPUTS:
        log(f"  [Diff] {exam_id}: chỉ {len(keep)}/{n_tests} test có sinh viên qua -> bỏ bằng chứng thực thi")
        return [None] * len(reports)
    if len(keep) < n_tests:
        log(f"  [Diff] {exam_id}: bỏ {n_tests - len(keep)} test không sinh viên nào qua")
    return [[r[j] for j in keep] if r else r for r in reports]


def summarize_report(report: Optional[List[dict]]) -> Optional[dict]:
    if not report:
        return None
    total = len(report)
    passed = sum(1 for t in report if t.get("passed"))
    lenient = sum(1 for t in report if t.get("lenient_passed", t.get("passed")))
    return {
        "total": total,
        "passed": passed,
        "rate": passed / total,
        "lenient_rate": lenient / total,
        "timeouts": sum(1 for t in report if t.get("timed_out")),
    }
