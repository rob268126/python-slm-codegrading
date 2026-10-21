"""
core/data.py
Load một split của dataset và dựng danh sách Submission dùng chung cho cả 3 task.

Cấu trúc split (train / public_test / private_test):
    exams.json, label_space.json, task1_grading.json, task2_error_taxonomy.json,
    task3_feedback.json, submissions/<exam_id>/<sample_id>.cpp
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.schema import DEFAULT_FEEDBACK_LEVEL, normalize_labels, parse_level

TASK_FILES = {
    1: "task1_grading.json",
    2: "task2_error_taxonomy.json",
    3: "task3_feedback.json",
}
EXAMS_FILE = "exams.json"

MULTI_PROBLEM = "multi_problem"
SINGLE_PROBLEM = "single_problem"

# g++ in lỗi với từ khoá "error" (vd "error:", "fatal error:"); tránh match "-Werror" hay tên biến.
_COMPILE_ERROR_RE = re.compile(r"(^|\s|:)(fatal )?error\s*:", re.IGNORECASE | re.MULTILINE)


def has_compile_error(compile_log: Optional[str]) -> bool:
    return bool(compile_log) and bool(_COMPILE_ERROR_RE.search(compile_log))


@dataclass
class Submission:
    sample_id: str
    exam_id: str
    exam_type: str
    statement: str
    code: str
    grading_policy: str = ""
    compile_log: str = ""
    test_report: Any = None
    feedback_level: int = DEFAULT_FEEDBACK_LEVEL
    # None = chưa biết (dataset không có compile_log và chưa chạy sandbox).
    compile_failed: Optional[bool] = None

    @property
    def is_multi_problem(self) -> bool:
        return self.exam_type == MULTI_PROBLEM


@dataclass
class SplitData:
    root: Path
    split: str
    exams: Dict[str, dict]
    tasks: Dict[int, dict] = field(default_factory=dict)

    def samples(self, task: int) -> List[dict]:
        return self.tasks.get(task, {}).get("samples", [])

    def sample_ids(self) -> List[str]:
        """Thứ tự sample_id chuẩn (lấy từ file task đầu tiên có mặt)."""
        for task in sorted(self.tasks):
            ids = [s["sample_id"] for s in self.samples(task)]
            if ids:
                return ids
        return []

    # ---------------- gold labels (chỉ có ở split train) ----------------
    def gold_rubrics(self) -> Dict[str, Dict[str, int]]:
        return {s["sample_id"]: s["output"]["rubric"]
                for s in self.samples(1) if s.get("output", {}).get("rubric")}

    def gold_taxonomy(self) -> Dict[str, List[str]]:
        gold = {}
        for s in self.samples(2):
            output = s.get("output")
            if output and "taxonomy_error" in output:
                gold[s["sample_id"]] = normalize_labels(output["taxonomy_error"])
        # Fallback: task3 train có taxonomy_error ở input.
        if not gold:
            for s in self.samples(3):
                if "taxonomy_error" in s.get("input", {}):
                    gold[s["sample_id"]] = normalize_labels(s["input"]["taxonomy_error"])
        return gold

    def gold_feedback(self) -> Dict[str, str]:
        return {s["sample_id"]: s["output"]["feedback"]
                for s in self.samples(3) if s.get("output", {}).get("feedback")}


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_split_root(dataset_root: str, split: str) -> Path:
    root = Path(dataset_root) / split
    return root if root.exists() else Path(dataset_root)


def load_split(dataset_root: str, split: str) -> SplitData:
    root = resolve_split_root(dataset_root, split)
    exams_json = _read_json(root / EXAMS_FILE) or {}
    exams = {e["exam_id"]: e for e in exams_json.get("exams", [])}
    tasks = {}
    for task, fname in TASK_FILES.items():
        data = _read_json(root / fname)
        if data is not None:
            tasks[task] = data
    if not tasks:
        raise FileNotFoundError(f"Không tìm thấy file task nào trong {root}")
    return SplitData(root=root, split=split, exams=exams, tasks=tasks)


def _read_code(root: Path, code_file: str) -> Optional[str]:
    candidates = [root / code_file, root / "submissions" / Path(code_file).name]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def build_submissions(data: SplitData) -> List[Submission]:
    """Gộp input của task1/2/3 theo sample_id thành danh sách Submission (thứ tự chuẩn)."""
    inputs: Dict[str, dict] = {}
    for task in sorted(data.tasks):
        for s in data.samples(task):
            merged = inputs.setdefault(s["sample_id"], {})
            for key, value in s.get("input", {}).items():
                merged.setdefault(key, value)

    submissions: List[Submission] = []
    for sid in data.sample_ids():
        inp = inputs[sid]
        exam = data.exams.get(inp.get("exam_id", ""), {})
        code = _read_code(data.root, inp.get("code_file", ""))
        if code is None:
            print(f"  [WARN] Không tìm thấy code của {sid} ({inp.get('code_file')}) -> dùng code rỗng")
            code = ""
        compile_log = inp.get("compile_log") or ""
        submissions.append(Submission(
            sample_id=sid,
            exam_id=inp.get("exam_id", ""),
            exam_type=inp.get("exam_type") or exam.get("exam_type", SINGLE_PROBLEM),
            statement=exam.get("statement", ""),
            grading_policy=exam.get("grading_policy", ""),
            code=code,
            compile_log=compile_log,
            test_report=inp.get("test_report"),
            feedback_level=parse_level(inp.get("target_feedback_level")),
            compile_failed=has_compile_error(compile_log) if compile_log else None,
        ))
    return submissions
