"""
sandbox/multi_harness.py
Kiểm thử thực thi cho đề multi_problem (bài nộp chỉ chứa hàm prob1..probK, không có main).

Quy trình hoàn toàn tự động, sinh từ chính metadata của đề (không viết tay theo đề, không dùng nhãn):
1. Đọc định nghĩa struct trong statement và prototype của từng câu (`exams.json -> problems[].prototype`).
2. Sinh hàm `main` gọi chuỗi P1 -> P2 -> P3/P4 trên file dữ liệu của đề (`data_file`) và in kết quả của một câu
   (struct được in đệ quy; danh sách liên kết / cây được duyệt theo con trỏ tự tham chiếu).
   Tham số chuỗi (vd. nationality, battle, *_year) lấy từ cột CSV có tên khớp với tên tham số.
3. Biên dịch harness cùng code sinh viên, chạy riêng từng câu (timeout -> vòng lặp vô hạn, exit != 0 -> crash).
4. Đáp án mỗi câu = output đồng thuận của các bài CÓ LÀM câu đó trong cùng đề (không cần lời giải tham chiếu).
   Câu không đạt đồng thuận đủ mạnh -> "unknown" (không dùng làm bằng chứng).
"""
from __future__ import annotations

import csv
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from inference.static_rules import function_bodies, strip_comments
from sandbox.cpp_compiler import CompiledProgram

RUN_TIMEOUT = 2.0
MAX_NODES = 2000
MIN_CONSENSUS = 3          # số bài tối thiểu cùng một output
MIN_CONSENSUS_SHARE = 0.2  # tỉ lệ tối thiểu trong các bài có làm câu đó và chạy xong

OK, WRONG, CRASH, TIMEOUT, MISSING, UNKNOWN = "ok", "wrong", "crash", "timeout", "missing", "unknown"

_CPP_BLOCK_RE = re.compile(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL)
_STRUCT_RE = re.compile(r"\bstruct\s+(\w+)\s*\{(.*?)\}\s*;", re.DOTALL)
_PROTO_RE = re.compile(r"^\s*([\w:<>\s]+?[\s*&]+)(prob\d+)\s*\(([^)]*)\)")
_EXC_RE = re.compile(r"invalid_argument|out_of_range|stoi|stof|stod|stol", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Phân tích đề
# ---------------------------------------------------------------------------
def parse_structs(statement: str) -> Dict[str, List[Tuple[str, str]]]:
    """{tên struct: [(kiểu, tên trường), ...]} theo thứ tự khai báo trong statement."""
    code = "\n".join(_CPP_BLOCK_RE.findall(statement)) or statement
    code = strip_comments(code)
    structs: Dict[str, List[Tuple[str, str]]] = {}
    for name, body in _STRUCT_RE.findall(code):
        fields = []
        for decl in body.split(";"):
            decl = " ".join(decl.split())
            if not decl:
                continue
            m = re.match(r"^([\w:]+(?:\s*<[^>]*>)?)\s*(.*)$", decl)
            if not m:
                continue
            base, rest = m.group(1).replace(" ", ""), m.group(2)
            for part in rest.split(","):
                part = part.strip()
                stars = len(part) - len(part.lstrip("*"))
                fname = part.lstrip("*").strip()
                if re.fullmatch(r"\w+", fname):
                    fields.append((base + "*" * stars, fname))
        structs[name] = fields
    return structs


def struct_definitions(statement: str) -> Dict[str, str]:
    """{tên struct: mã định nghĩa} nguyên văn từ statement (dùng bù cho bài nộp thiếu struct)."""
    code = strip_comments("\n".join(_CPP_BLOCK_RE.findall(statement)) or statement)
    return {m.group(1): m.group(0) for m in _STRUCT_RE.finditer(code)}


def parse_prototype(proto: str) -> Optional[dict]:
    m = _PROTO_RE.match(proto.split("//")[0])
    if not m:
        return None
    ret = " ".join(m.group(1).split()).replace(" *", "*").replace(" &", "&")
    params = []
    for p in [x.strip() for x in m.group(3).split(",") if x.strip()]:
        pm = re.match(r"^(.*?)([\w]+)$", p)
        if pm:
            params.append((" ".join(pm.group(1).split()).replace(" *", "*").replace(" &", "&"), pm.group(2)))
    return {"ret": ret, "name": m.group(2), "params": params}


def _node_struct(structs: Dict[str, List[Tuple[str, str]]]) -> Optional[str]:
    for name, fields in structs.items():
        if any(t == f"{name}*" for t, _ in fields):
            return name
    return None


# ---------------------------------------------------------------------------
# Chọn giá trị tham số chuỗi từ file dữ liệu
# ---------------------------------------------------------------------------
def _tokens(text: str) -> set:
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).replace("_", " ").lower().split()
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


_YEAR_RE = re.compile(r"(\d{1,4})(?:\s*(AD|BC))?\s*$")


def _read_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    return (rows[0], rows[1:]) if rows else ([], [])


def _year_of(cell: str) -> Optional[Tuple[str, int]]:
    m = _YEAR_RE.search(cell.strip())
    if not m:
        return None
    year = int(m.group(1))
    era = m.group(2)
    text = f"{year} {era}" if era else str(year)
    return text, (-year if era == "BC" else year)


def string_arg_candidates(params: Sequence[Tuple[str, str]], data_path: Path) -> List[List[str]]:
    """Các bộ giá trị cho những tham số `string` (trừ filename) của một câu, lấy từ cột CSV khớp tên tham số."""
    names = [n for t, n in params if t.startswith("string") and "file" not in n.lower()]
    if not names:
        return [[]]
    header, rows = _read_csv(data_path)
    date_cols = [j for j in range(len(header))
                 if rows and sum(bool(re.search(r"\d+-\d+-\d+", r[j])) for r in rows if j < len(r)) > len(rows) // 2]

    def column_for(param: str) -> Optional[int]:
        ptok = _tokens(param) - {"year"}
        scored = [(len(ptok & _tokens(h)), -j, j) for j, h in enumerate(header)]
        best = max(scored, default=(0, 0, None))
        if best[0] > 0:
            return best[2]
        return date_cols[0] if ("year" in param.lower() and date_cols) else None

    def values(param: str) -> List[str]:
        j = column_for(param)
        if j is None:
            return []
        cells = [r[j] for r in rows if j < len(r)]
        if "year" in param.lower():
            years = sorted({y for y in (_year_of(c) for c in cells) if y}, key=lambda y: y[1])
            return [y[0] for y in years]
        items = Counter(x.strip() for c in cells for x in re.split(r"[|;]", c) if x.strip())
        return [v for v, _ in items.most_common()]

    vals = {n: values(n) for n in names}
    if any(not v for v in vals.values()):
        return []
    year_params = [n for n in names if "year" in n.lower()]
    if len(year_params) == 2 and len(names) == 2:  # khoảng [start, end]
        ys = vals[year_params[0]]
        q = lambda f: ys[min(len(ys) - 1, int(f * (len(ys) - 1)))]
        return [[q(0.25), q(0.75)], [ys[0], ys[-1]], [q(0.5), q(0.5)]]
    picks = []
    for k in range(3):  # 3 bộ: giá trị phổ biến nhất, nhì, và một giá trị ở giữa danh sách
        row = []
        for n in names:
            v = vals[n]
            idx = [0, 1, len(v) // 2][k]
            row.append(v[min(idx, len(v) - 1)])
        if row not in picks:
            picks.append(row)
    return picks


# ---------------------------------------------------------------------------
# Sinh harness C++
# ---------------------------------------------------------------------------
_OPEN, _CLOSE = "@@<", ">@@"
_PREAMBLE = """#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdlib>
using namespace std;
"""


def _cpp_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _dump_functions(structs: Dict[str, List[Tuple[str, str]]], node: Optional[str]) -> str:
    plain = [s for s in structs if s != node]
    lines = ["static void H_dump(ostream& o, const string& x) { o << '\"' << x << '\"'; }",
             "static void H_dump(ostream& o, int x) { o << x; }",
             "static void H_dump(ostream& o, long long x) { o << x; }",
             "static void H_dump(ostream& o, double x) { o << x; }",
             "static void H_dump(ostream& o, float x) { o << x; }",
             "static void H_dump(ostream& o, bool x) { o << x; }",
             "template <class T> static void H_dump(ostream& o, const vector<T>& v) {"
             " o << '['; for (size_t i = 0; i < v.size() && i < 100000; ++i) { if (i) o << ','; H_dump(o, v[i]); } o << ']'; }"]
    lines += [f"static void H_dump(ostream& o, const {s}& x);" for s in plain]
    for s in plain:
        body = []
        for t, f in structs[s]:
            if t.endswith("*"):
                continue  # mảng cấp phát động không rõ độ dài -> bỏ qua
            body.append(f'o << "{f}="; H_dump(o, x.{f}); o << ";";')
        lines.append(f"static void H_dump(ostream& o, const {s}& x) {{ o << '{{'; {' '.join(body)} o << '}}'; }}")
    if node:
        ptrs = [f for t, f in structs[node] if t == f"{node}*"]
        vals = [f for t, f in structs[node] if t != f"{node}*" and not t.endswith("*")]
        val_dump = " ".join(f"H_dump(o, p->{f});" for f in vals)
        if len(ptrs) == 1:
            lines.append(
                f"static void H_dumpNode(ostream& o, {node}* p) {{ int c = 0; o << \"LIST\\n\";"
                f" while (p && c < {MAX_NODES}) {{ {val_dump} o << '\\n'; p = p->{ptrs[0]}; ++c; }}"
                f" if (p) o << \"...CYCLE\"; o << \"END\\n\"; }}")
        else:
            rec = " ".join(f"H_tree(o, p->{f}, c);" for f in ptrs)
            lines.append(
                f"static void H_tree(ostream& o, {node}* p, int& c) {{ if (!p) {{ o << '#'; return; }}"
                f" if (++c > {MAX_NODES}) {{ o << \"...CYCLE\"; return; }} o << '('; {val_dump} {rec} o << ')'; }}")
            lines.append(f"static void H_dumpNode(ostream& o, {node}* p) {{ int c = 0; o << \"TREE\\n\"; H_tree(o, p, c); o << \"\\nEND\\n\"; }}")
    return "\n".join(lines)


def build_harness(exam: dict, data_name: str, data_path: Path) -> Optional[dict]:
    """Trả về {"main": mã C++ của main, "pids": [...]} hoặc None nếu đề không đủ thông tin để sinh tự động."""
    structs = parse_structs(exam.get("statement", ""))
    protos = [parse_prototype(p.get("prototype", "")) for p in exam.get("problems", [])]
    if not structs or not protos or any(p is None for p in protos):
        return None
    node = _node_struct(structs)
    pids = [p.get("pid", f"P{i + 1}") for i, p in enumerate(exam["problems"])]

    produced: Dict[str, str] = {}  # kiểu (bỏ &) -> biến chứa kết quả câu trước
    size_var: Optional[str] = None
    blocks: List[str] = []
    setup: List[str] = []          # lời gọi các câu trước (để dựng dữ liệu cho câu k)
    for k, proto in enumerate(protos):
        args, pre, ref_out = [], [], None
        for t, n in proto["params"]:
            base = t.rstrip("&").strip()
            if base == "string" and "file" in n.lower():
                args.append(_cpp_str(data_name))
            elif base == "int" and t.endswith("&"):
                pre.append(f"int H_n{k} = 0;")
                args.append(f"H_n{k}")
                size_var = f"H_n{k}"
            elif base == "int":
                if size_var is None:
                    return None
                args.append(size_var)
            elif base in produced:
                if t.endswith("&"):
                    pre.append(f"{base} H_ref{k} = {produced[base]};")
                    args.append(f"H_ref{k}")
                    ref_out = f"H_ref{k}"
                else:
                    args.append(produced[base])
            elif base == "string":
                args.append(None)  # điền từ CSV
            else:
                return None
        string_slots = [i for i, a in enumerate(args) if a is None]
        cands = string_arg_candidates(proto["params"], data_path) if string_slots else [[]]
        if string_slots and not cands:
            return None

        ret = proto["ret"].strip()
        var = f"H_r{k}"
        calls = []
        for cand in cands:
            filled = list(args)
            for slot, val in zip(string_slots, cand):
                filled[slot] = _cpp_str(val)
            call = f"{proto['name']}({', '.join(filled)})"
            if ret == "void":
                stmt = f"{call};"
                out = f"H_dumpNode(cout, {ref_out});" if (ref_out and node) else ""
            else:
                stmt = f"{ret} {var} = {call};"
                if ret.rstrip("*") == node and ret.endswith("*"):
                    out = f"H_dumpNode(cout, {var});"
                elif ret.endswith("*") and size_var:
                    out = (f"cout << \"N=\" << {size_var} << '\\n'; if ({size_var} >= 0 && {size_var} <= 100000 && {var})"
                           f" for (int i = 0; i < {size_var}; ++i) {{ H_dump(cout, {var}[i]); cout << '\\n'; }}")
                elif ret.startswith("vector<") and ret[7:-1] in structs:
                    out = (f"cout << \"N=\" << {var}.size() << '\\n';"
                           f" for (size_t i = 0; i < {var}.size() && i < 100000; ++i) {{ H_dump(cout, {var}[i]); cout << '\\n'; }}")
                elif ret.endswith("*"):
                    return None
                else:
                    out = f"H_dump(cout, {var}); cout << '\\n';"
            if out:  # phần harness in nằm giữa hai mốc, tách khỏi những gì code sinh viên tự in
                out = f"cout << \"\\n{_OPEN}\\n\"; {out} cout << \"\\n{_CLOSE}\\n\";"
            # Mỗi bộ tham số dựng lại dữ liệu từ đầu (P4 sửa danh sách tại chỗ).
            calls.append("{ " + " ".join(setup + pre) + f" {stmt} {out} cout.flush(); }}")
        blocks.append(f"  if (H_k == {k}) {{ {' '.join(calls)} return 0; }}")

        # Kết quả câu k (không cần tham số chuỗi) dùng làm đầu vào cho các câu sau -> dựng sẵn.
        # Không gọi câu không ai cần (vd. P3 khi chạy P4) để lỗi của câu đó không lan sang câu khác.
        later_types = {t.rstrip("&").strip() for p in protos[k + 1:] for t, _ in p["params"]}
        if ret != "void" and not string_slots and ret in later_types:
            setup.extend(pre)
            setup.append(f"{ret} H_s{k} = {proto['name']}({', '.join(args)});")
            produced[ret] = f"H_s{k}"

    main = (_dump_functions(structs, node) + "\nint main(int argc, char** argv) {\n"
            "  int H_k = argc > 1 ? atoi(argv[1]) : 0;\n" + "\n".join(blocks) + "\n  return 0;\n}\n")
    return {"main": main, "pids": pids, "protos": protos, "struct_defs": struct_definitions(exam.get("statement", ""))}


# ---------------------------------------------------------------------------
# Chạy và so khớp đồng thuận
# ---------------------------------------------------------------------------
def _attempted(code: str, fname: str) -> bool:
    """Câu có được làm không: thân hàm còn câu lệnh ngoài `return <mặc định>;`."""
    fn = function_bodies(code).get(fname)
    if fn is None:
        return False
    body = re.sub(r"\s+", " ", fn["body"]).strip()
    body = re.sub(r"^return\s*[^;]*;$", "", body).strip()
    return len(body) > 0


_FIELD_SPLIT_RE = re.compile(r"[;\n]")
NEAR_MIN_FIELDS = 30       # chỉ so khớp gần đúng với output dài (danh sách bản ghi)
NEAR_MAX_MISMATCH = 0.10   # tối đa 10% trường lệch


def outputs_match(actual: str, expected: str) -> bool:
    """Khớp chính xác; hoặc với output dài (nhiều bản ghi): cùng cấu trúc, lệch không quá 10% số trường.

    Một trường phụ đọc nhầm cột (vd. số trận) không làm hỏng cả câu, nhưng sai số bản ghi / thứ tự /
    nhiều trường thì vẫn là sai.
    """
    if actual == expected:
        return True
    a, e = _FIELD_SPLIT_RE.split(actual), _FIELD_SPLIT_RE.split(expected)
    if len(e) < NEAR_MIN_FIELDS or len(a) != len(e):
        return False
    if actual.splitlines()[:1] != expected.splitlines()[:1]:  # dòng đầu (N=, LIST, TREE) phải khớp
        return False
    mismatch = sum(x != y for x, y in zip(a, e))
    return mismatch <= NEAR_MAX_MISMATCH * len(e)


def _harness_segments(stdout: str) -> str:
    """Chỉ giữ phần do harness in (giữa hai mốc); bỏ mọi thứ code sinh viên tự in ra."""
    stdout = stdout.replace("\r\n", "\n")  # stdout chế độ text trên Windows
    segs =re.findall(re.escape(_OPEN) + r"\n(.*?)\n" + re.escape(_CLOSE), stdout, re.DOTALL)
    return "\n".join(s.strip() for s in segs)


def count_data_rows(data_path: Path) -> int:
    _header, rows = _read_csv(data_path)
    return sum(1 for r in rows if any(c.strip() for c in r))


def record_count(output: str) -> Optional[int]:
    """Số bản ghi trong dump của câu đọc file ("N=<n>" + n dòng bản ghi); None nếu không hợp lệ."""
    lines = output.splitlines()
    if not lines or not lines[0].startswith("N="):
        return None
    try:
        n = int(lines[0][2:])
    except ValueError:
        return None
    records = sum(1 for l in lines[1:] if l.startswith("{"))
    return n if records == n else None


def _reads_file(proto: dict) -> bool:
    return any(t.rstrip("&").strip() == "string" and "file" in n.lower() for t, n in proto["params"])


def _is_empty_structure(output: Optional[str]) -> bool:
    """Danh sách / cây rỗng không được dùng làm đáp án đồng thuận (thường là kết quả của bài làm dở)."""
    if output is None:
        return False
    return ("LIST" in output or "TREE" in output) and "{" not in output


def _default_return(ret: str) -> str:
    if ret == "void":
        return ""
    if ret.endswith("*"):
        return "return 0;"
    return f"return {ret}();"


def _assemble(student_code: str, harness: dict) -> str:
    """Code sinh viên + struct còn thiếu (lấy từ đề) + hàm rỗng cho câu không làm + main của harness.

    - main tự viết của sinh viên (nếu còn) được đổi tên để không trùng với main của harness.
    - Câu không làm (không có định nghĩa probK) được thay bằng hàm trả về giá trị mặc định -> verdict "missing".
    """
    code = (student_code or "").replace("﻿", "")
    plain = strip_comments(code)
    defined = set(function_bodies(code))
    have_structs = set(re.findall(r"\bstruct\s+(\w+)\s*\{", plain))
    extra = [text for name, text in harness["struct_defs"].items() if name not in have_structs]
    stubs = []
    for p in harness["protos"]:
        if p["name"] not in defined:
            params = ", ".join(f"{t} {n}" for t, n in p["params"])
            stubs.append(f"{p['ret']} {p['name']}({params}) {{ {_default_return(p['ret'])} }}")
    return (_PREAMBLE + "#define main H_student_main\n" + code + "\n#undef main\n"
            + "\n".join(extra) + "\n" + "\n".join(stubs) + "\n" + harness["main"])


class MultiProblemTester:
    def __init__(self, run_timeout: float = RUN_TIMEOUT, log: Callable[[str], None] = print):
        self.run_timeout = run_timeout
        self.log = log

    def run_student(self, code: str, harness: dict, data_path: Path, data_name: str) -> Optional[List[dict]]:
        """Kết quả thô từng câu: {status: ok-run|crash|timeout, output, stderr, attempted}. None nếu harness không biên dịch được."""
        with CompiledProgram(_assemble(code, harness)) as prog:
            if not prog.ok:
                return None
            shutil.copy(data_path, Path(prog.workdir) / data_name)
            raw = []
            for k, pid in enumerate(harness["pids"]):
                r = prog.run("", self.run_timeout, args=[str(k)])
                status = "timeout" if r["timed_out"] else ("run" if r["returncode"] == 0 else "crash")
                raw.append({"pid": pid, "status": status, "output": _harness_segments(r["output"]),
                            "stderr": r["stderr"][-400:], "attempted": _attempted(code, f"prob{k + 1}")})
            return raw

    def evaluate_exam(self, exam: dict, codes: Dict[str, str], data_path: Path) -> Dict[str, List[dict]]:
        """{sample_id: [{pid, verdict, detail}, ...]} cho các bài chạy được harness."""
        exam_id = exam.get("exam_id", "")
        harness = build_harness(exam, data_path.name, data_path)
        if harness is None:
            self.log(f"  [Multi] {exam_id}: không sinh được harness từ đề -> bỏ qua")
            return {}
        raws = {sid: self.run_student(code, harness, data_path, data_path.name) for sid, code in codes.items()}
        ran = {sid: r for sid, r in raws.items() if r is not None}
        self.log(f"  [Multi] {exam_id}: harness biên dịch được với {len(ran)}/{len(codes)} bài")

        # Câu k chỉ lấy đồng thuận trên các bài đã đúng câu tiên quyết của nó: bài sai P1 dựng ra dữ liệu
        # rỗng/giống nhau ở các câu sau và sẽ chiếm đa số nếu được tính.
        prereqs = {pid: [p["pid"] for p in exam.get("problems", []) if pid in (p.get("is_prerequisite_for") or [])]
                   for pid in harness["pids"]}
        pid_idx = {pid: i for i, pid in enumerate(harness["pids"])}
        verdicts: Dict[str, Dict[str, str]] = {sid: {} for sid in ran}
        expected: List[Optional[str]] = []
        n_rows = count_data_rows(data_path)
        for k, pid in enumerate(harness["pids"]):
            if _reads_file(harness["protos"][k]):
                # Câu đọc file: đề yêu cầu đọc TOÀN BỘ bản ghi -> chấm khách quan bằng số bản ghi trả về,
                # không dựa vào đồng thuận (nhiều bài cùng sai một kiểu, vd. giả định n cho sẵn -> N=0).
                expected.append(None)
                for sid, raw in ran.items():
                    v = self._verdict(raw[k], None)
                    if v == UNKNOWN:
                        v = OK if record_count(raw[k]["output"]) == n_rows else WRONG
                    verdicts[sid][pid] = v
                n_ok = sum(verdicts[s][pid] == OK for s in ran)
                self.log(f"  [Multi] {exam_id} {pid}: {n_ok}/{len(ran)} bài đọc đủ {n_rows} bản ghi")
                continue
            eligible =[sid for sid in ran if all(verdicts[sid].get(q) == OK for q in prereqs[pid] if q in pid_idx)]
            outs = Counter(ran[sid][k]["output"] for sid in eligible
                           if ran[sid][k]["attempted"] and ran[sid][k]["status"] == "run" and ran[sid][k]["output"])
            done = sum(outs.values())
            # Đáp án = output có nhiều bài "gần giống" nhất (gần giống: xem outputs_match).
            support = {o: sum(c for o2, c in outs.items() if outputs_match(o2, o)) for o in outs}
            value, cnt = max(support.items(), key=lambda kv: (kv[1], outs[kv[0]])) if support else (None, 0)
            ok = cnt >= MIN_CONSENSUS and cnt >= MIN_CONSENSUS_SHARE * done and not _is_empty_structure(value)
            expected.append(value if ok else None)
            self.log(f"  [Multi] {exam_id} {pid}: đồng thuận {cnt}/{done} bài" + ("" if ok else " -> không đủ, bỏ qua"))
            for sid, raw in ran.items():
                verdicts[sid][pid] = self._verdict(raw[k], expected[k])

        report: Dict[str, List[dict]] = {}
        for sid, raw in ran.items():
            report[sid] = [{"pid": r["pid"], "verdict": verdicts[sid][r["pid"]],
                            "exception": bool(r["status"] == "crash" and _EXC_RE.search(r["stderr"]))}
                           for r in raw]
        return report

    @staticmethod
    def _verdict(r: dict, expected: Optional[str]) -> str:
        if r["status"] == "timeout":
            return TIMEOUT
        if not r["attempted"]:
            return MISSING
        if r["status"] == "crash":
            return CRASH
        if expected is None:
            return UNKNOWN
        return OK if outputs_match(r["output"], expected) else WRONG
