"""
inference/feedback_gen.py
Task 3: Sinh phản hồi tiếng Việt có kiểm soát mức độ sư phạm (Level 1 - 4).

- `build_feedback_prompt` + `LEVEL_GUIDELINES`: prompt dùng để tạo dữ liệu fine-tune LoRA (giữ nguyên).
- `FeedbackGenerator`: lúc inference, phản hồi được ghép từ bằng chứng khách quan (log g++, test thực thi)
  + câu chẩn đoán của base model (tắt LoRA) đã qua kiểm tra chặt; Level 1 dùng template không nêu vị trí/code.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from core.data import Submission
from core.schema import (
    DEFAULT_FEEDBACK_LEVEL,
    LABEL_ALGORITHM,
    LABEL_ARRAY,
    LABEL_COMPILE,
    LABEL_EDGE,
    LABEL_FUNCTION,
    LABEL_HARDCODE,
    LABEL_IO,
    LABEL_LOGIC,
    LABEL_LOOP,
    LABEL_STYLE,
    normalize_labels,
)

# Định nghĩa quy tắc 4 cấp độ phản hồi theo label_space.json
LEVEL_GUIDELINES: Dict[int, str] = {
    1: "Chỉ báo có vấn đề / khen ngợi, không chỉ ra vị trí lỗi. CẤM: không nêu nguyên nhân cụ thể, KHÔNG đưa code sửa.",
    2: "Chỉ ra lỗi ở đâu và nguyên nhân là gì. CẤM: không đưa cách sửa cụ thể, KHÔNG đưa code.",
    3: "Chẩn đoán + gợi ý hướng sửa. Có thể có snippet nhỏ. CẤM: không đưa lời giải hoàn chỉnh.",
    4: "Được phép đưa code sửa / lời giải mẫu hoàn chỉnh.",
}


def build_feedback_prompt(statement: str, code: str, level: int, errors: Sequence[str],
                          few_shot: str = "", warning_note: str = "") -> str:
    """Prompt Task 3 dùng chung cho inference và fine-tune để hai bên khớp nhau tuyệt đối."""
    level_rule = LEVEL_GUIDELINES.get(level, LEVEL_GUIDELINES[DEFAULT_FEEDBACK_LEVEL])
    err_display = ", ".join(errors) if errors else "Không có lỗi"
    return (
        f"<|im_start|>system\n"
        f"Bạn là giảng viên lập trình C++ đang chấm bài cho sinh viên.\n"
        f"Nhiệm vụ: Viết đoạn nhận xét ngắn gọn, đúng trọng tâm bằng tiếng Việt cho sinh viên.\n"
        f"Mức phản hồi yêu cầu: Level {level}.\n"
        f"Quy tắc cấp độ: {level_rule}\n"
        f"{warning_note}"
        f"Tuyệt đối tuân thủ quy tắc sư phạm trên.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{few_shot}"
        f"### Bài cần chấm:\n"
        f"Đề bài:\n{statement[:800]}\n\n"
        f"Code sinh viên:\n```cpp\n{code[:1500]}\n```\n\n"
        f"Các lỗi chẩn đoán: {err_display}\n"
        f"Yêu cầu: Viết trực tiếp nhận xét Level {level} bằng tiếng Việt, không lặp lại tiền tố:\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )



# ---------------------------------------------------------------------------
# Inference Task 3: phản hồi bám bằng chứng (compiler + test thực thi + nhãn Task 2)
# ---------------------------------------------------------------------------
# Câu chẩn đoán nguyên nhân (Level 2+) và gợi ý hướng sửa bằng lời (chỉ Level 3+), không chứa code.
_CAUSE: Dict[str, str] = {
    LABEL_COMPILE: "Chương trình không biên dịch được nên không chạy được test nào.",
    LABEL_IO: "Định dạng nhập/xuất chưa đúng yêu cầu của đề.",
    LABEL_LOGIC: "Cách xử lý/tính toán chưa đúng với yêu cầu đề bài nên kết quả bị sai.",
    LABEL_LOOP: "Vòng lặp có điều kiện dừng hoặc việc cập nhật biến lặp chưa đúng.",
    LABEL_ARRAY: "Phần xử lý mảng/chuỗi chưa đúng (chỉ số hoặc kích thước).",
    LABEL_FUNCTION: "Hàm được cài đặt hoặc gọi chưa đúng (tham số hoặc giá trị trả về).",
    LABEL_EDGE: "Chương trình chưa xử lý đúng các trường hợp biên của đề.",
    LABEL_ALGORITHM: "Thuật toán được chọn chưa giải đúng bài toán.",
    LABEL_HARDCODE: "Bài làm gán cứng giá trị thay vì xử lý theo dữ liệu đầu vào.",
    LABEL_STYLE: "Cách trình bày code chưa rõ ràng.",
}
_HINT: Dict[str, str] = {
    LABEL_COMPILE: "Em đọc kỹ thông báo của compiler tại dòng báo lỗi và sửa lại cú pháp tương ứng.",
    LABEL_IO: "Em so sánh từng ký tự output với ví dụ trong đề (khoảng trắng, xuống dòng, chữ hoa/thường).",
    LABEL_LOGIC: "Em thử chạy tay chương trình với input bị sai để tìm bước tính chưa đúng.",
    LABEL_LOOP: "Em kiểm tra lại điều kiện dừng và việc cập nhật biến lặp sau mỗi vòng.",
    LABEL_ARRAY: "Em kiểm tra lại giới hạn chỉ số và kích thước của mảng/chuỗi.",
    LABEL_FUNCTION: "Em kiểm tra lại kiểu trả về, tham số và chỗ gọi hàm.",
    LABEL_EDGE: "Em thử thêm các trường hợp đặc biệt như giá trị nhỏ nhất, lớn nhất hoặc dữ liệu rỗng.",
    LABEL_ALGORITHM: "Em xem lại ý tưởng thuật toán trên giấy trước khi cài đặt.",
    LABEL_HARDCODE: "Em xử lý theo dữ liệu đọc vào thay vì gán sẵn giá trị.",
    LABEL_STYLE: "Em đặt tên biến rõ nghĩa và tách code thành các hàm nhỏ.",
}
# Level 1: chỉ báo có vấn đề ở mảng nào, không chỉ vị trí/nguyên nhân cụ thể.
_AREA: Dict[str, str] = {
    LABEL_IO: "định dạng nhập/xuất",
    LABEL_LOGIC: "cách xử lý của chương trình",
    LABEL_LOOP: "các vòng lặp",
    LABEL_ARRAY: "phần xử lý mảng/chuỗi",
    LABEL_FUNCTION: "các hàm",
    LABEL_EDGE: "các trường hợp đặc biệt",
    LABEL_ALGORITHM: "thuật toán",
    LABEL_HARDCODE: "việc gán cứng giá trị",
    LABEL_STYLE: "cách trình bày code",
}

_LLM_LEVEL_RULE: Dict[int, str] = {
    2: "Mức Level 2 - Chẩn đoán: chỉ ra lỗi nằm ở đâu và nguyên nhân là gì. CẤM đưa cách sửa cụ thể, CẤM viết code.",
    3: "Mức Level 3 - Gợi ý sửa: chỉ ra lỗi ở đâu, nguyên nhân và gợi ý hướng sửa bằng lời. CẤM viết code, CẤM đưa lời giải hoàn chỉnh.",
    4: "Mức Level 4 - Lời giải: chỉ ra lỗi, nguyên nhân và cách sửa bằng lời ngắn gọn.",
}

_COMPILE_ERR_RE = re.compile(r"[\w./\\-]+\.(?:cpp|h|hpp|cc):(\d+):(?:\d+:)?\s*(?:fatal )?error:\s*([^\n]+)")
_GENERIC_ERR_RE = re.compile(r"(?:fatal )?error:\s*([^\n]+)")
_VIET_RE = re.compile(r"[ăâđêôơưàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹ]", re.IGNORECASE)
_CODE_RE = re.compile(
    r"`|#\s*include|\bcout\s*<<|\bcin\s*>>|[{}]|==|!=|<=|>=|\+\+|&&|\|\||->|"
    r"\b(?:int|long|double|float|char|bool|void|string|auto)\s+\w+\s*[=;,(\[]|"
    r"\b(?:for|while|if|switch)\s*\(|\breturn\b[^.\n]*;|\b\w+\s*\[\s*\w*\s*\]\s*=|\b\w+\s*[-+*/]?=\s*[\w(\"']"
)
_FIX_RE = re.compile(
    r"sửa (?:lại )?thành|thay (?:\S+ ){0,3}bằng|đổi (?:\S+ ){0,3}thành|nên dùng|nên sử dụng|hãy dùng|"
    r"hãy sử dụng|cần thêm|hãy thêm|chỉ cần|cách sửa|để sửa|sửa lại|sửa như",
    re.IGNORECASE,
)
_PRAISE_ONLY_RE = re.compile(
    r"không (?:có|phát hiện|tìm thấy|thấy) lỗi|đúng (?:hết|toàn bộ|tất cả)|hoàn toàn (?:đúng|chính xác)|^\W*tốt\W*$",
    re.IGNORECASE,
)
_LEAD_RE = re.compile(r"^\s*[-*•]?\s*(?:nhận xét|phản hồi|chẩn đoán|feedback)\s*:\s*", re.IGNORECASE)
_COPY_RATIO = 0.6


def _one_line(text: str, limit: int = 60) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _labels_text(labels: Sequence[str]) -> str:
    return ", ".join(label[0].lower() + label[1:] for label in labels)


def _compile_errors(compile_log: str, limit: int = 2) -> List[str]:
    """Trích 'dòng N: <thông báo>' từ log g++ (bỏ đường dẫn file tạm và dòng code bị trích)."""
    found = [f"dòng {line}: {msg.strip()}" for line, msg in _COMPILE_ERR_RE.findall(compile_log or "")]
    if not found:
        found = [m.strip() for m in _GENERIC_ERR_RE.findall(compile_log or "")]
    unique: List[str] = []
    for item in found:
        item = _one_line(re.sub(r"\s*\[-[\w=+-]+\]\s*$", "", item), 140)
        if item not in unique:
            unique.append(item)
    return unique[:limit]


def collect_evidence(sub: Submission) -> dict:
    """Bằng chứng khách quan đã có sau bước chấm (compile_log + test_report), không gọi lại sandbox."""
    report = sub.test_report if isinstance(sub.test_report, list) else None
    ev = {"compile_errors": _compile_errors(sub.compile_log), "tests": None}
    if report:
        total = len(report)
        lenient = [t for t in report if t.get("lenient_passed", t.get("passed"))]
        exact = [t for t in report if t.get("passed")]
        wrong = next((t for t in report if not t.get("lenient_passed", t.get("passed"))), None)
        bad_format = next((t for t in report if t.get("lenient_passed") and not t.get("passed")), None)
        ev["tests"] = {
            "total": total,
            "lenient": len(lenient),
            "exact": len(exact),
            "timeout": next((t for t in report if t.get("timed_out")), None),
            "wrong": wrong,
            "bad_format": bad_format,
        }
    return ev


def _evidence_sentences(ev: dict, compile_failed: bool) -> List[str]:
    """Câu mô tả bằng chứng cho Level 2+ (vị trí lỗi biên dịch / test sai), không chứa code sửa."""
    if compile_failed:
        if ev["compile_errors"]:
            return [f"Lỗi biên dịch: {'; '.join(ev['compile_errors'])}."]
        return ["Chương trình không biên dịch được."]
    tests = ev["tests"]
    if not tests:
        return []
    out: List[str] = []
    if tests["timeout"] is not None:
        out.append(f"Chương trình chạy quá thời gian (không dừng) với input \"{_one_line(tests['timeout']['input'])}\".")
    if tests["wrong"] is not None and tests["wrong"] is not tests["timeout"]:
        w = tests["wrong"]
        out.append(
            f"Chương trình sai {tests['total'] - tests['lenient']}/{tests['total']} test tự động, ví dụ với input "
            f"\"{_one_line(w['input'])}\" cần in \"{_one_line(w['expected'])}\" nhưng chương trình in \"{_one_line(w['actual'])}\"."
        )
    elif tests["wrong"] is None and tests["bad_format"] is not None:
        b = tests["bad_format"]
        out.append(
            f"Kết quả tính đúng nhưng output chưa đúng định dạng đề yêu cầu, ví dụ với input \"{_one_line(b['input'])}\" "
            f"cần in \"{_one_line(b['expected'])}\" nhưng chương trình in \"{_one_line(b['actual'])}\"."
        )
    return out


def _all_tests_passed(ev: dict) -> bool:
    tests = ev["tests"]
    return bool(tests) and tests["exact"] == tests["total"]


def build_diagnosis_prompt(sub: Submission, statement: str, code: str, level: int,
                           labels: Sequence[str], evidence: Sequence[str]) -> str:
    """Prompt cho base model (tắt LoRA): chỉ ra vị trí + nguyên nhân dựa trên code có đánh số dòng."""
    task = "chỉ ra lỗi nằm ở đâu (tên hàm, số dòng hoặc câu) và nguyên nhân"
    if level >= 3:
        task += ", kèm gợi ý hướng sửa bằng lời"
    ev_text = "\n".join(f"- {e}" for e in evidence) if evidence else "- Không có test tự động cho bài này."
    return (
        "<|im_start|>system\n"
        "Bạn là giảng viên lập trình C++ đang nhận xét bài làm của sinh viên, xưng \"em\" với sinh viên.\n"
        f"{_LLM_LEVEL_RULE.get(level, _LLM_LEVEL_RULE[2])}\n"
        "Chỉ viết lời nhận xét tiếng Việt. Không viết code, không dùng ký tự `, không trích nguyên câu lệnh.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Đề bài:\n{statement}\n\n"
        f"Code sinh viên (đầu mỗi dòng là số dòng):\n{code}\n\n"
        f"Kết quả kiểm tra tự động:\n{ev_text}\n"
        f"Nhãn lỗi đã chẩn đoán: {_labels_text(labels)}\n\n"
        f"Viết 1-2 câu {task}. Không lặp lại kết quả kiểm tra, không nói bài làm đúng.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _number_lines(code: str) -> str:
    return "\n".join(f"{i:3d}| {line}" for i, line in enumerate((code or "").splitlines(), 1))


class FeedbackGenerator:
    """Task 3: phản hồi tiếng Việt bám bằng chứng thực thi, kiểm soát Level bằng template + kiểm tra chặt.

    - Level 1 và bài không có lỗi: template (Level 1 không được nêu vị trí/nguyên nhân -> LLM không thêm giá trị).
    - Level 2+: câu bằng chứng (dòng lỗi biên dịch, test sai) + 1-2 câu vị trí/nguyên nhân do base model viết
      (tắt LoRA để không chép lại phản hồi train). Câu LLM bị loại nếu chứa code, cách sửa (Level 2),
      khen sai khi có lỗi, hoặc gần trùng phản hồi train -> dùng câu nguyên nhân theo nhãn.
    """

    def __init__(self, llm=None, train_feedback_file: Optional[str] = None):
        self.llm = llm
        self.train_feedbacks: List[str] = []
        if train_feedback_file and Path(train_feedback_file).exists():
            data = json.loads(Path(train_feedback_file).read_text(encoding="utf-8"))
            self.train_feedbacks = [
                s.get("output", {}).get("feedback", "").strip() for s in data.get("samples", [])
            ]
            self.train_feedbacks = [fb for fb in self.train_feedbacks if len(fb) >= 20]
            print(f"  [FeedbackGen] Nạp {len(self.train_feedbacks)} phản hồi train để chặn sao chép.")

    # ---------------- kiểm tra câu do LLM sinh ----------------
    def _is_copy(self, text: str) -> bool:
        low = text.lower()
        return any(SequenceMatcher(None, low, fb.lower()).ratio() >= _COPY_RATIO for fb in self.train_feedbacks)

    def _clean_llm(self, text: str, level: int, has_errors: bool) -> str:
        """Trả về câu chẩn đoán hợp lệ hoặc chuỗi rỗng nếu không dùng được."""
        text = (text or "").split("<|")[0].strip()
        text = text.split("\n\n")[0].strip()
        text = _LEAD_RE.sub("", " ".join(text.split())).strip().strip('"').strip()
        if len(text) > 400:
            cut = text[:400].rfind(".")
            text = text[: cut + 1] if cut > 50 else ""
        if len(text) < 15 or not _VIET_RE.search(text) or _CODE_RE.search(text):
            return ""
        if level <= 2 and _FIX_RE.search(text):
            return ""
        if has_errors and _PRAISE_ONLY_RE.search(text):
            return ""
        if self._is_copy(text):
            return ""
        return text

    def _llm_diagnosis(self, sub: Submission, level: int, labels: Sequence[str], evidence: Sequence[str]) -> str:
        if self.llm is None or not (sub.code or "").strip():
            return ""
        prompt = self.llm.fit_prompt(
            lambda s, c: build_diagnosis_prompt(sub, s, c, level, labels, evidence),
            sub.statement, _number_lines(sub.code),
        )
        response = self.llm.generate(prompt, max_new_tokens=160, use_adapter=False)
        return self._clean_llm(response, level, bool(labels))

    # ---------------- ghép phản hồi theo Level ----------------
    def generate_one(self, sub: Submission, errors: Sequence[str], level: Optional[int] = None) -> str:
        level = level if level in (1, 2, 3, 4) else sub.feedback_level
        labels = normalize_labels(errors)
        ev = collect_evidence(sub)
        compile_failed = LABEL_COMPILE in labels

        if not labels:
            if level == 1:
                return "Tốt."
            if _all_tests_passed(ev):
                return "Tốt. Chương trình cho kết quả đúng với tất cả test kiểm tra."
            return "Tốt. Bài làm đáp ứng yêu cầu của đề."

        if labels == [LABEL_STYLE]:
            note = "Tốt, nhưng em chú ý trình bày code rõ ràng hơn (lỗi style)."
            return note if level == 1 else f"{note} {_HINT[LABEL_STYLE]}"

        if level == 1:
            if compile_failed:
                return "Bài làm còn lỗi biên dịch, chương trình chưa chạy được. Em xem lại bài nhé."
            areas = [_AREA[l] for l in labels if l in _AREA][:3]
            areas_text = ", ".join(areas[:-1]) + f" và {areas[-1]}" if len(areas) > 1 else areas[0]
            return f"Bài làm còn {_labels_text(labels)}. Em kiểm tra lại {areas_text} nhé."

        evidence = _evidence_sentences(ev, compile_failed)
        diagnosis = self._llm_diagnosis(sub, level, labels, evidence)
        parts = list(evidence)
        if diagnosis:
            parts.append(diagnosis)
        else:
            parts += [_CAUSE[l] for l in labels if not (l == LABEL_COMPILE and evidence)][:2]
        if level >= 3 and not (diagnosis and level == 3 and _FIX_RE.search(diagnosis)):
            parts += [_HINT[l] for l in labels][:2]
        parts.append(f"(Chẩn đoán: {_labels_text(labels)}.)")
        return " ".join(p for p in parts if p)
