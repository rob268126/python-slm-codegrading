"""
core/schema.py
Single Source of Truth cho Rubric, Taxonomy lỗi và Mức phản hồi sư phạm.
Sử dụng Pydantic v2 để tự động hóa:
- Type safety & Schema validation nghiêm ngặt theo đúng chuẩn BTC (label_space.json).
- Tự động clamp giá trị về đúng miền (ge, le) và ép kiểu (coercion) an toàn từ LLM.
- Tự động áp dụng ràng buộc tiên quyết (prerequisite: compilable=0 => logic=0, edge_case=0, complexity=0).
- Tự động map và lọc nhãn taxonomy về strictly 10 nhãn chính thức.
- Cấu trúc xuất file nộp (Task 1, 2, 3) ngăn chặn 100% lỗi từ chối submission trên Leaderboard.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Rubric Constants
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

# Khi bài không biên dịch được, giảng viên cho 0 ở mọi chiều thuật toán/biên/phức tạp
DIMS_ZEROED_WHEN_NOT_COMPILABLE: Tuple[str, ...] = ("io_format", "logic", "edge_case", "complexity")

# ---------------------------------------------------------------------------
# Taxonomy Constants (10 nhãn chính thức, đúng thứ tự trong label_space.json)
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

OfficialTaxonomyLabel = Literal[
    "Lỗi biên dịch", "Lỗi nhập/xuất", "Lỗi logic", "Lỗi vòng lặp", "Lỗi mảng/chuỗi",
    "Lỗi hàm", "Lỗi edge case", "Lỗi thuật toán", "Lỗi hard-code", "Lỗi style"
]

# LoRA adapter v3 được train trên taxonomy legacy (sinh tổng hợp trong notebook cũ).
# Map các nhãn legacy đó về nhãn chính thức; giá trị None = loại bỏ.
LEGACY_LABEL_MAP: Dict[str, Optional[str]] = {
    "Lỗi cú pháp": LABEL_COMPILE,
    "Lỗi syntax": LABEL_COMPILE,
    "Lỗi kiểu dữ liệu": LABEL_COMPILE,
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
# Feedback levels Constants
# ---------------------------------------------------------------------------
FEEDBACK_LEVELS: Tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_FEEDBACK_LEVEL = 2
_LEVEL_RE = re.compile(r"(\d)")


# ---------------------------------------------------------------------------
# Normalization & Parsing Helpers
# ---------------------------------------------------------------------------
def normalize_labels(labels: Optional[Iterable]) -> List[str]:
    """Map nhãn bất kỳ (chính thức / legacy / sai hoa-thường) về tập 10 nhãn chính thức.

    Nhãn không nhận diện được bị loại bỏ. Kết quả không trùng lặp và theo đúng thứ tự chuẩn.
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


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class Rubric(BaseModel):
    """Rubric 6 chiều theo chuẩn chấm thi.

    Tự động:
    - Coerce từ float/string sang int.
    - Clamp giá trị về đúng miền [min, max] của từng chiều.
    - Áp dụng prerequisite policy: compilable=0 => io_format=0, logic=0, edge_case=0, complexity=0.
    """
    model_config = ConfigDict(extra="ignore")

    compilable: int = Field(default=0, ge=0, le=1)
    io_format: int = Field(default=0, ge=0, le=1)
    logic: int = Field(default=0, ge=0, le=4)
    edge_case: int = Field(default=0, ge=0, le=2)
    complexity: int = Field(default=0, ge=0, le=1)
    code_quality: int = Field(default=0, ge=0, le=1)

    @field_validator("*", mode="before")
    @classmethod
    def _clamp_and_coerce(cls, v, info):
        dim = info.field_name
        if dim in RUBRIC_RANGES:
            lo, hi = RUBRIC_RANGES[dim]
            try:
                v = int(round(float(v)))
            except (TypeError, ValueError):
                v = lo
            return max(lo, min(hi, v))
        return v

    @model_validator(mode="after")
    def _enforce_prerequisites(self) -> "Rubric":
        if self.compilable == 0:
            self.io_format = 0
            self.logic = 0
            self.edge_case = 0
            self.complexity = 0
        return self

    @property
    def total_score(self) -> int:
        return (
            self.compilable + self.io_format + self.logic +
            self.edge_case + self.complexity + self.code_quality
        )

    def to_dict(self) -> Dict[str, int]:
        return {d: getattr(self, d) for d in RUBRIC_DIMS}


def clamp_rubric(raw: Optional[Mapping], default: Optional[Mapping] = None) -> Dict[str, int]:
    """Ép rubric về đúng 6 chiều, kiểu int và nằm trong miền giá trị hợp lệ qua Pydantic."""
    merged = dict(default or {})
    if isinstance(raw, Mapping):
        merged.update(raw)
    return Rubric.model_validate(merged).to_dict()


def total_score(rubric: Mapping[str, int]) -> int:
    """Tính tổng điểm rubric an toàn qua Pydantic."""
    return Rubric.model_validate(rubric).total_score


class GradingResult(BaseModel):
    """Kết quả chấm điểm thống nhất cho Task 1 & Task 2."""
    model_config = ConfigDict(extra="ignore")

    sample_id: str
    rubric: Rubric = Field(default_factory=Rubric)
    taxonomy_error: List[str] = Field(default_factory=list)
    total_score: int = 0
    rationale: str = ""
    compiler_verified: bool = False

    @field_validator("rubric", mode="before")
    @classmethod
    def _coerce_rubric(cls, v):
        if isinstance(v, Rubric):
            return v
        if isinstance(v, Mapping):
            return Rubric.model_validate(v)
        return Rubric()

    @field_validator("taxonomy_error", mode="before")
    @classmethod
    def _coerce_taxonomy(cls, v):
        return normalize_labels(v)

    @model_validator(mode="after")
    def _sync_total(self) -> "GradingResult":
        self.total_score = self.rubric.total_score
        return self

    def apply_compiler_ground_truth(self, compile_failed: bool) -> "GradingResult":
        """Đồng bộ ground truth từ g++ compiler vào rubric và taxonomy."""
        if compile_failed:
            self.rubric.compilable = 0
            self.rubric.io_format = 0
            self.rubric.logic = 0
            self.rubric.edge_case = 0
            self.rubric.complexity = 0
            if LABEL_COMPILE not in self.taxonomy_error:
                self.taxonomy_error = normalize_labels([LABEL_COMPILE] + list(self.taxonomy_error))
        else:
            if self.rubric.compilable == 0:
                self.rubric.compilable = 1
            if LABEL_COMPILE in self.taxonomy_error:
                self.taxonomy_error = [e for e in self.taxonomy_error if e != LABEL_COMPILE]

        self.total_score = self.rubric.total_score
        self.compiler_verified = True
        return self


# ---------------------------------------------------------------------------
# Leaderboard Export Submission Models
# Đảm bảo 100% khớp schema BTC yêu cầu cho file predictions.json
# ---------------------------------------------------------------------------
class Task1Output(BaseModel):
    rubric: Dict[str, int]
    total_score: int


class Task1Submission(BaseModel):
    sample_id: str
    output: Task1Output


class Task2Output(BaseModel):
    taxonomy_error: List[str]

    @field_validator("taxonomy_error", mode="before")
    @classmethod
    def _validate_tax(cls, v):
        return normalize_labels(v)


class Task2Submission(BaseModel):
    sample_id: str
    output: Task2Output


class Task3Output(BaseModel):
    taxonomy_error: List[str]
    feedback: str

    @field_validator("taxonomy_error", mode="before")
    @classmethod
    def _validate_tax(cls, v):
        return normalize_labels(v)


class Task3Submission(BaseModel):
    sample_id: str
    output: Task3Output
