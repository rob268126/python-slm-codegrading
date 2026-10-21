"""
core/schema.py
Single source of truth cho rubric, taxonomy lỗi và mức phản hồi.
Phải khớp với `label_space.json` của BTC — leaderboard từ chối mọi nhãn ngoài danh sách này.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------
RUBRIC_RANGES: Dict[str, Tuple[int, int]] = {
    "compilable": (0, 1),
    "io_format": (0, 1),
    "logic": (0, 4),
    "edge_case": (0, 2),
    "complexity": (0, 1),
    "code_quality": (0, 1),
}
RUBRIC_DIMS: Tuple[str, ...] = tuple(RUBRIC_RANGES)
TOTAL_RANGE: Tuple[int, int] = (0, 10)

# Khi bài không biên dịch được, giảng viên cho 0 ở mọi chiều trừ code_quality
# (6/6 mẫu compilable=0 trong tập train đều tuân theo quy luật này).
DIMS_ZEROED_WHEN_NOT_COMPILABLE: Tuple[str, ...] = ("io_format", "logic", "edge_case", "complexity")

# ---------------------------------------------------------------------------
# Taxonomy lỗi (10 nhãn chính thức, đúng thứ tự trong label_space.json)
# ---------------------------------------------------------------------------
LABEL_COMPILE = "Lỗi biên dịch"
LABEL_IO = "Lỗi nhập/xuất"
LABEL_LOGIC = "Lỗi logic"
LABEL_LOOP = "Lỗi vòng lặp"
LABEL_ARRAY = "Lỗi mảng/chuỗi"
LABEL_FUNCTION = "Lỗi hàm"
LABEL_EDGE = "Lỗi edge case"
LABEL_ALGORITHM = "Lỗi thuật toán"
LABEL_HARDCODE = "Lỗi hard-code"
LABEL_STYLE = "Lỗi style"

TAXONOMY_LABELS: Tuple[str, ...] = (
    LABEL_COMPILE, LABEL_IO, LABEL_LOGIC, LABEL_LOOP, LABEL_ARRAY,
    LABEL_FUNCTION, LABEL_EDGE, LABEL_ALGORITHM, LABEL_HARDCODE, LABEL_STYLE,
)

# LoRA adapter v3 được train trên taxonomy "legacy" (sinh tổng hợp trong notebook).
# Map các nhãn legacy đó về nhãn chính thức; giá trị None = bỏ nhãn.
LEGACY_LABEL_MAP: Dict[str, Optional[str]] = {
    "Lỗi cú pháp": LABEL_COMPILE,
    "Lỗi syntax": LABEL_COMPILE,
    "Lỗi kiểu dữ liệu": LABEL_COMPILE,  # notebook sinh nhãn này từ compiler error "cannot convert"
    "Lỗi biến/hàm chưa khai báo": LABEL_COMPILE,
    "Lỗi liên kết (missing main)": LABEL_COMPILE,
    "Lỗi biên dịch khác": LABEL_COMPILE,
    "Lỗi I/O": LABEL_IO,
    "Lỗi IO": LABEL_IO,
    "Lỗi xử lý biên": LABEL_EDGE,
    "Lỗi bộ nhớ": LABEL_ARRAY,
    "Lỗi logic sai": LABEL_LOGIC,
    "Không có lỗi": None,
    "No error": None,
}

_CANONICAL_BY_KEY: Dict[str, Optional[str]] = {
    **{label.casefold(): label for label in TAXONOMY_LABELS},
    **{legacy.casefold(): official for legacy, official in LEGACY_LABEL_MAP.items()},
}

# ---------------------------------------------------------------------------
# Feedback levels
# ---------------------------------------------------------------------------
FEEDBACK_LEVELS: Tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_FEEDBACK_LEVEL = 2
_LEVEL_RE = re.compile(r"(\d)")


def normalize_labels(labels: Optional[Iterable]) -> List[str]:
    """Map nhãn bất kỳ (chính thức / legacy / sai hoa-thường) về tập 10 nhãn chính thức.

    Nhãn không nhận diện được bị loại bỏ. Kết quả không trùng lặp và theo thứ tự chuẩn.
    """
    if labels is None:
        return []
    if isinstance(labels, str):
        labels = [labels]
    found = set()
    for raw in labels:
        if not isinstance(raw, str):
            continue
        official = _CANONICAL_BY_KEY.get(raw.strip().casefold())
        if official:
            found.add(official)
    return [label for label in TAXONOMY_LABELS if label in found]


def clamp_rubric(raw: Optional[Mapping], default: Optional[Mapping] = None) -> Dict[str, int]:
    """Ép rubric về đúng 6 chiều, kiểu int và nằm trong miền giá trị hợp lệ."""
    raw = raw if isinstance(raw, Mapping) else {}
    default = default or {}
    clean: Dict[str, int] = {}
    for dim, (lo, hi) in RUBRIC_RANGES.items():
        value = raw.get(dim, default.get(dim, 0))
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            value = int(default.get(dim, 0))
        clean[dim] = max(lo, min(hi, value))
    return clean


def total_score(rubric: Mapping[str, int]) -> int:
    return int(sum(rubric.get(dim, 0) for dim in RUBRIC_DIMS))


def parse_level(value, default: int = DEFAULT_FEEDBACK_LEVEL) -> int:
    """'Level 2 - Chẩn đoán' | 2 | '2' -> 2. Giá trị không hợp lệ -> default."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        level = value
    else:
        match = _LEVEL_RE.search(str(value or ""))
        level = int(match.group(1)) if match else default
    return level if level in FEEDBACK_LEVELS else default
