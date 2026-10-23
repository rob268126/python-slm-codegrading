"""
inference/feedback_gen.py
Task 3: Sinh phản hồi tiếng Việt có kiểm soát mức độ sư phạm (Level 1 - 4).

1. Generator: Sinh phản hồi sư phạm theo Level với few-shot từ tập train.
2. Kiểm tra Level 1 & 2 bằng regex (`is_level_compliant`); nếu vi phạm thì sinh lại 1 lần với cảnh báo.
3. Enforcer: Lớp bảo vệ cuối cùng đảm bảo tuân thủ Level Compliance.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from core.data import Submission
from core.schema import (
    DEFAULT_FEEDBACK_LEVEL,
    LABEL_COMPILE,
    normalize_labels,
    parse_level,
)
from evaluation.metrics import is_level_compliant

# Định nghĩa quy tắc 4 cấp độ phản hồi theo label_space.json
LEVEL_GUIDELINES: Dict[int, str] = {
    1: "Chỉ báo có vấn đề / khen ngợi, không chỉ ra vị trí lỗi. CẤM: không nêu nguyên nhân cụ thể, KHÔNG đưa code sửa.",
    2: "Chỉ ra lỗi ở đâu và nguyên nhân là gì. CẤM: không đưa cách sửa cụ thể, KHÔNG đưa code.",
    3: "Chẩn đoán + gợi ý hướng sửa. Có thể có snippet nhỏ. CẤM: không đưa lời giải hoàn chỉnh.",
    4: "Được phép đưa code sửa / lời giải mẫu hoàn chỉnh.",
}

# Regex làm sạch code khi sinh phản hồi cho Level 1 & 2
_MD_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`\n]{15,}`")
_FORBIDDEN_PHRASES = [
    r"lời giải mẫu là:?",
    r"code sửa lại như sau:?",
    r"dưới đây là code hoàn chỉnh:?",
    r"bạn có thể thay bằng:?",
]


def enforce_compliance(feedback: str, level: int) -> str:
    """Đảm bảo phản hồi tuân thủ Level mà không để lại chuỗi thô kệch."""
    clean = feedback.strip()
    # Loại bỏ tiền tố nhại lại từ prompt ví dụ như "- Lỗi: ... \n Phản hồi: "
    if "phản hồi:" in clean.lower():
        parts = re.split(r"phản hồi\s*:\s*", clean, flags=re.IGNORECASE)
        if len(parts) > 1 and parts[-1].strip():
            clean = parts[-1].strip()
    clean = re.sub(r"^[-*•]?\s*(Lỗi|Phản hồi|Nhận xét|Chẩn đoán)\s*:\s*", "", clean, flags=re.IGNORECASE)

    if level <= 2:
        # Xóa triệt để các code block và inline code dài
        clean = _MD_CODE_BLOCK_RE.sub("", clean)
        clean = _INLINE_CODE_RE.sub("", clean)
        for p in _FORBIDDEN_PHRASES:
            clean = re.sub(p, "", clean, flags=re.IGNORECASE)
        # Làm sạch khoảng trắng thừa
        clean = re.sub(r"\n\s*\n\s*\n+", "\n\n", clean).strip()

    if not clean or clean.lower() in ("tốt.", "tốt", "ok"):
        if level == 1:
            clean = "Bài làm có một số điểm chưa tối ưu, em hãy đọc kỹ lại yêu cầu đề bài và kiểm tra các trường hợp biên."
        elif level == 2:
            clean = "Em cần kiểm tra lại logic xử lý thuật toán và điều kiện dừng của chương trình."
        else:
            clean = "Em hãy kiểm tra lại cấu trúc vòng lặp và xử lý các trường hợp biên của bài toán."
    return clean


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


class FeedbackGenerator:
    """Bộ sinh phản hồi sư phạm tiếng Việt có kiểm soát Level."""

    def __init__(self, llm=None, train_feedback_file: Optional[str] = None):
        """
        Args:
            llm: Instance của inference.llm.LLM (hoặc None để dùng heuristic).
            train_feedback_file: Đường dẫn file dataset/train/task3_feedback.json để nạp few-shot examples.
        """
        self.llm = llm
        self.few_shot_by_level: Dict[int, List[dict]] = {1: [], 2: [], 3: [], 4: []}
        if train_feedback_file and Path(train_feedback_file).exists():
            self._load_train_examples(train_feedback_file)

    def _load_train_examples(self, filepath: str):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            for s in data.get("samples", []):
                lvl = parse_level(s.get("input", {}).get("target_feedback_level"))
                fb = s.get("output", {}).get("feedback", "").strip()
                errs = s.get("input", {}).get("taxonomy_error", [])
                exam_type = s.get("input", {}).get("exam_type", "")
                if fb:
                    self.few_shot_by_level[lvl].append({
                        "errors": errs,
                        "feedback": fb,
                        "exam_type": exam_type,
                    })
            total = sum(len(v) for v in self.few_shot_by_level.values())
            print(f"  [FeedbackGen] Đã nạp {total} mẫu few-shot từ tập train Task 3.")
        except Exception as e:
            print(f"  [FeedbackGen] Không thể đọc few-shot data: {e}")

    def _get_few_shot_block(self, level: int, exam_type: str) -> str:
        """Lấy 1-2 mẫu few-shot phù hợp với level và loại bài thi."""
        candidates = self.few_shot_by_level.get(level, [])
        if not candidates:
            return ""

        # Ưu tiên mẫu cùng exam_type
        matched = [c for c in candidates if c.get("exam_type") == exam_type]
        chosen = (matched or candidates)[:2]

        lines = ["### Mẫu nhận xét tham khảo của giảng viên:"]
        for ex in chosen:
            lines.append(f"- Nhận xét: \"{ex['feedback'].strip()}\"")
        lines.append("")
        return "\n".join(lines)

    def heuristic_feedback(self, sub: Submission, errors: Sequence[str], level: int) -> str:
        """Sinh phản hồi sư phạm chuẩn mực không cần LLM."""
        norm_errors = normalize_labels(errors)

        if not norm_errors:
            if level == 1:
                return "Bài làm rất tốt, code trình bày rõ ràng và đáp ứng đầy đủ yêu cầu."
            elif level == 2:
                return "Chương trình chạy tốt, không phát hiện lỗi logic hay biên dịch."
            else:
                return "Bài làm chính xác. Em có thể tối ưu thêm về cấu trúc code hoặc cách đặt tên biến để code đẹp hơn."

        has_compile_err = LABEL_COMPILE in norm_errors
        err_str = ", ".join(norm_errors)

        if has_compile_err:
            if level == 1:
                return "Chương trình chưa thể biên dịch thành công. Em hãy đọc kỹ thông báo từ compiler để chỉnh sửa."
            elif level == 2:
                return f"Bài làm mắc {LABEL_COMPILE}. Em cần kiểm tra lại cú pháp khai báo và thư viện trước khi nộp."
            elif level == 3:
                return f"Lỗi biên dịch: Em hãy kiểm tra các dấu chấm phẩy, khai báo biến và include đầy đủ các header cần thiết."
            else:
                return f"Chương trình gặp lỗi biên dịch. Em cần sửa lại cú pháp hàm và bổ sung các header tương ứng."

        if level == 1:
            return f"Bài làm còn tồn tại một số điểm chưa chính xác ({err_str}). Em hãy rà soát lại giải thuật và các trường hợp biên."
        elif level == 2:
            return f"Chương trình mắc các lỗi sau: {err_str}. Em hãy xem lại cách cài đặt thuật toán và điều kiện vòng lặp."
        elif level == 3:
            return f"Phát hiện lỗi: {err_str}. Gợi ý: Em cần kiểm tra lại giá trị khởi tạo của biến, các trường hợp mảng rỗng hoặc chỉ số vượt biên."
        else:
            return f"Phát hiện lỗi: {err_str}. Hãy kiểm tra cẩn thận logic rẽ nhánh và các trường hợp biên của bài toán."

    def generate_one(self, sub: Submission, errors: Sequence[str], level: Optional[int] = None) -> str:
        """Sinh phản hồi: Generator -> kiểm tra regex (retry 1 lần nếu vi phạm) -> Enforcer."""
        target_level = level if level in (1, 2, 3, 4) else sub.feedback_level
        norm_errors = normalize_labels(errors)

        if self.llm is None:
            raw_fb = self.heuristic_feedback(sub, norm_errors, target_level)
            return enforce_compliance(raw_fb, target_level)

        few_shot = self._get_few_shot_block(target_level, sub.exam_type)

        def _render(stmt: str, code: str, warning_note: str = "") -> str:
            return build_feedback_prompt(stmt, code, target_level, norm_errors, few_shot, warning_note)

        prompt = self.llm.fit_prompt(lambda s, c: _render(s, c), sub.statement, sub.code)
        response = self.llm.generate(prompt, max_new_tokens=256)

        if target_level <= 2:
            if not is_level_compliant(response, target_level):
                # Kích hoạt Self-Refine retry với warning nghiêm khắc
                warning = "CẢNH BÁO: Lần sinh trước đã bị vi phạm do chứa code. Tuyệt đối KHÔNG viết bất kỳ đoạn code nào, chỉ dùng lời giải thích tiếng Việt!\n"
                retry_prompt = self.llm.fit_prompt(lambda s, c: _render(s, c, warning), sub.statement, sub.code)
                response = self.llm.generate(retry_prompt, max_new_tokens=256)

        # Safeguard chẩn đoán: Nếu bài có lỗi nhưng mô hình lại nhại theo few-shot sinh "không có lỗi" hoặc "tốt"
        fb_lower = response.lower()
        if norm_errors and ("không phát hiện lỗi" in fb_lower or "không có lỗi" in fb_lower or response.strip().lower() in ("tốt.", "tốt")):
            response = self.heuristic_feedback(sub, norm_errors, target_level)

        return enforce_compliance(response, target_level)
