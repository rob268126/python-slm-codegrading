"""
inference/static_rules.py
Luật phân tích tĩnh mã C++ để bắt các nhãn taxonomy hiếm (Task 2/3) mà LLM greedy hầu như không sinh ra.

Mỗi luật được rút ra từ phản hồi giảng viên trong tập train (xem docstring từng hàm) và chỉ bắn khi tín hiệu rõ ràng
(ưu tiên precision): Macro-F1 trung bình đều trên 10 nhãn nên mỗi nhãn hiếm bắt đúng vài bài đã đáng kể.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

from core.schema import LABEL_ARRAY, LABEL_HARDCODE, LABEL_LOOP

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
_STRUCT_RE = re.compile(r"\b(?:struct|class)\s+(\w+)\s*\{")
_FUNC_HEAD_RE = re.compile(r"\b([\w:<>]+(?:\s*[*&]+)?)\s*[*&]?\s*\b(\w+)\s*\(([^;{}()]*)\)\s*(?:const\s*)?\{")
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "else", "do"}


def strip_comments(code: str) -> str:
    return _COMMENT_RE.sub(" ", code or "")


def _strip_strings(code: str) -> str:
    return _STRING_RE.sub('""', code)


def _match_brace(text: str, open_idx: int) -> int:
    """Vị trí dấu '}' khớp với '{' tại open_idx (text đã bỏ comment/chuỗi); -1 nếu không khớp."""
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def function_bodies(code: str) -> Dict[str, dict]:
    """{tên hàm: {"ret": kiểu trả về, "body": thân hàm}} cho các hàm định nghĩa ở mức ngoài cùng."""
    text = _strip_strings(strip_comments(code))
    funcs: Dict[str, dict] = {}
    pos = 0
    while True:
        m = _FUNC_HEAD_RE.search(text, pos)
        if not m:
            break
        name = m.group(2)
        open_idx = m.end() - 1
        close = _match_brace(text, open_idx)
        if close < 0:
            break
        if name not in _KEYWORDS:
            head = text[m.start():open_idx]
            funcs.setdefault(name, {"ret": head.split(name)[0], "body": text[open_idx + 1:close]})
            pos = close + 1
        else:
            pos = m.end()
    return funcs


def struct_types(code: str) -> Set[str]:
    return set(_STRUCT_RE.findall(strip_comments(code)))


# ---------------------------------------------------------------------------
# Lỗi vòng lặp: duyệt danh sách liên kết / cây mà không dịch con trỏ -> lặp vô tận
# (train EX01-S106 "không có cur = cur->next", EX01-S107 "không có head = head->next").
# ---------------------------------------------------------------------------
_WHILE_RE = re.compile(r"\bwhile\s*\(")
_PTR_COND_RE = re.compile(r"^\s*!?\s*\(?\s*(\w+)\s*(?:->\s*\w+\s*)?(?:!=\s*(?:NULL|nullptr|0)\s*)?\)?\s*$")


def _paren_span(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def pointer_loop_without_advance(code: str) -> bool:
    text = _strip_strings(strip_comments(code))
    for m in _WHILE_RE.finditer(text):
        close_paren = _paren_span(text, m.end() - 1)
        if close_paren < 0:
            continue
        cond = text[m.end():close_paren]
        after = text[close_paren + 1:].lstrip()
        if not after.startswith("{"):
            continue  # do-while hoặc thân một dòng: bỏ qua để giữ precision
        cm = _PTR_COND_RE.match(cond)
        if not cm:
            continue
        var = cm.group(1)
        if not re.search(rf"\b{var}\s*->", text):
            continue  # biến điều kiện không phải con trỏ node
        open_idx = close_paren + 1 + (len(text[close_paren + 1:]) - len(after))
        close = _match_brace(text, open_idx)
        if close < 0:
            continue
        body = text[open_idx + 1:close]
        advanced = re.search(rf"\b{var}\s*=(?!=)|\b{var}\s*(?:\+\+|--)|(?:\+\+|--)\s*{var}\b", body)
        escapes = re.search(r"\b(?:break|return|goto|throw)\b", body)
        # Truyền biến vào hàm theo tham chiếu (vd. advance(cur)) cũng có thể dịch con trỏ -> bỏ qua.
        # (chỉ tính khi truyền nguyên biến, không tính `cur->field`).
        passed = re.search(rf"[(,]\s*&?\s*{var}\s*[,)]", body)
        if not advanced and not escapes and not passed:
            return True
    return False


# ---------------------------------------------------------------------------
# Lỗi mảng/chuỗi: trả về địa chỉ mảng cục bộ (train EX01-S105 "đọc vào 1 mảng tĩnh rồi return địa chỉ
# phần tử đầu tiên -> ra khỏi hàm là mảng không còn").
# ---------------------------------------------------------------------------
def returns_local_array(code: str) -> bool:
    for fn in function_bodies(code).values():
        if "*" not in fn["ret"]:
            continue
        body = fn["body"]
        arrays = re.findall(r"(?<![\w:])(?!static\b|return\b|new\b)(\w+)\s+(\w+)\s*\[[^\]]+\]\s*;", body)
        for _typ, name in arrays:
            alias = re.findall(rf"(\w+)\s*=\s*&?\s*{name}\b(?:\s*\[\s*0\s*\])?\s*;", body)
            targets = [name] + alias
            for t in targets:
                if re.search(rf"\breturn\s+&?\s*{t}\b(?:\s*\[\s*0\s*\])?\s*;", body):
                    return True
    return False


# ---------------------------------------------------------------------------
# Lỗi hard-code (multi_problem):
#   - include / gán cứng tên file dữ liệu trong hàm (train EX01-S103, EX01-S112)
#   - cấp phát mảng struct với kích thước hằng thay vì đếm số dòng (train EX01-S108 `new General[21]`)
# ---------------------------------------------------------------------------
def hardcoded_data(code: str) -> bool:
    plain = strip_comments(code)
    if re.search(r'#\s*include\s*"[^"\n]*\.(?:csv|txt)"', plain, re.IGNORECASE):
        return True
    funcs = function_bodies(code)
    # Thân hàm đã bỏ chuỗi -> tìm tên file trên code gốc nhưng loại phần main (main tự test được phép).
    body_src = re.split(r"\bint\s+main\s*\(", plain, maxsplit=1)[0]
    if re.search(r'"[^"\n]*\.csv"', body_src, re.IGNORECASE):
        return True
    types = struct_types(code)
    if not types:
        return False
    type_alt = "|".join(re.escape(t) for t in types)
    for name, fn in funcs.items():
        if name == "main":
            continue
        body = fn["body"]
        if re.search(rf"\bnew\s+(?:{type_alt})\s*\[\s*\d{{2,}}\s*\]", body):
            return True
        if re.search(rf"\b(?:{type_alt})\s+\w+\s*\[\s*\d{{2,}}\s*\]", body) and "*" in fn["ret"]:
            return True
    return False


def static_labels(code: str, is_multi_problem: bool) -> List[str]:
    """Nhãn taxonomy bắt được bằng phân tích tĩnh (không phụ thuộc compiler/LLM)."""
    if not (code or "").strip():
        return []
    labels: List[str] = []
    if pointer_loop_without_advance(code):
        labels.append(LABEL_LOOP)
    if returns_local_array(code):
        labels.append(LABEL_ARRAY)
    if is_multi_problem and hardcoded_data(code):
        labels.append(LABEL_HARDCODE)
    return labels
