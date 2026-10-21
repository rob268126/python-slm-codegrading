"""
sandbox/cpp_compiler.py
C++ Compiler Sandbox — sinh compile_log thực tế bằng g++ khi dataset không cung cấp.

Cờ biên dịch bám theo `exams.json -> note` của BTC:
  - single_problem : g++ -std=c++11            (biên dịch + link đầy đủ)
  - multi_problem  : g++ -std=c++11 -fsyntax-only   (bài nộp chỉ chứa hàm, không có main)

Dataset README cảnh báo một số bài viết theo MSVC (`void main`, dùng `pow` không include <cmath>)
vẫn được giảng viên chấm compilable=1. `msvc_compat=True` mô phỏng độ "dễ dãi" đó để giảm báo lỗi giả.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from typing import List, Optional

MULTI_PROBLEM = "multi_problem"

_WINDOWS_GPP_PATHS = (
    r"C:\msys64\ucrt64\bin\g++.exe",
    r"C:\msys64\mingw64\bin\g++.exe",
    r"C:\MinGW\bin\g++.exe",
    r"C:\TDM-GCC-64\bin\g++.exe",
    r"C:\Program Files\mingw64\bin\g++.exe",
)
_VOID_MAIN_RE = re.compile(r"\bvoid\s+main\s*\(")
_MSVC_ONLY_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"](stdafx\.h|pch\.h|conio\.h)[>"].*$', re.MULTILINE | re.IGNORECASE)

# Phân loại lỗi g++ (để phân tích lỗi / ablation, không dùng làm nhãn).
_ERROR_PATTERNS = (
    (("undeclared", "not declared"), "Lỗi biến/hàm chưa khai báo"),
    (("no match for", "no matching function"), "Lỗi hàm"),
    (("cannot convert", "invalid conversion"), "Lỗi kiểu dữ liệu"),
    (("redeclaration", "redefinition"), "Lỗi khai báo trùng"),
    (("undefined reference", "ld returned"), "Lỗi liên kết"),
    (("no such file",), "Lỗi thiếu header file"),
    (("expected",), "Lỗi cú pháp"),
)


@lru_cache(maxsize=1)
def find_gcc() -> Optional[str]:
    """Tìm đường dẫn g++ (PATH trước, sau đó các vị trí cài đặt phổ biến trên Windows)."""
    found = shutil.which("g++")
    if found:
        return found
    return next((p for p in _WINDOWS_GPP_PATHS if os.path.exists(p)), None)


def _apply_msvc_compat(code: str) -> str:
    code = _VOID_MAIN_RE.sub("int main(", code)
    return _MSVC_ONLY_INCLUDE_RE.sub("", code)


def _classify_errors(error_lines: List[str]) -> List[str]:
    types = []
    for line in error_lines:
        low = line.lower()
        label = next((name for keys, name in _ERROR_PATTERNS if any(k in low for k in keys)), "Lỗi biên dịch khác")
        types.append(label)
    return types


def _empty_result() -> dict:
    return {
        "compilable": False,
        "compile_success": False,
        "compile_log": "",
        "errors": [],
        "warnings": [],
        "error_types": [],
    }


def _binary_name(workdir: str) -> str:
    return os.path.join(workdir, "prog.exe" if os.name == "nt" else "prog")


def _compile_in(workdir: str, code: str, exam_type: str, std: str, timeout: int, msvc_compat: bool) -> dict:
    result = _empty_result()
    gpp = find_gcc()
    if gpp is None:
        result["compile_log"] = "g++ not found. Please install MinGW/g++ and add to PATH."
        result["error_types"].append("Lỗi môi trường")
        return result

    src = os.path.join(workdir, "student.cpp")
    with open(src, "w", encoding="utf-8") as f:
        f.write(_apply_msvc_compat(code) if msvc_compat else code)

    cmd = [gpp, f"-std={std}"]
    if msvc_compat:
        cmd += ["-include", "cmath"]
    cmd += ["-fsyntax-only", src] if exam_type == MULTI_PROBLEM else ["-o", _binary_name(workdir), src]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        result["compile_log"] = "COMPILATION TIMEOUT"
        result["errors"].append("Compilation timed out")
        result["error_types"].append("Lỗi timeout")
        return result

    # Chuẩn hoá đường dẫn file tạm để log ổn định giữa các lần chạy.
    log = proc.stderr.replace(src, "student.cpp").strip()
    lines = [l.strip() for l in log.splitlines() if l.strip()]
    result["compile_log"] = log
    result["compile_success"] = result["compilable"] = proc.returncode == 0
    result["errors"] = [l for l in lines if "error" in l.lower() and "warning:" not in l.lower()]
    result["warnings"] = [l for l in lines if "warning:" in l.lower()]
    result["error_types"] = _classify_errors(result["errors"])
    return result


def compile_cpp(code: str, exam_type: str = "single_problem", std: str = "c++11",
                timeout: int = 15, msvc_compat: bool = True) -> dict:
    """Biên dịch code C++ và trả về dict:
    compilable, compile_success, compile_log, errors, warnings, error_types.
    """
    with tempfile.TemporaryDirectory() as workdir:
        return _compile_in(workdir, code, exam_type, std, timeout, msvc_compat)


def run_cpp_with_input(code: str, test_input: str = "", timeout: int = 5, std: str = "c++11",
                       msvc_compat: bool = True) -> dict:
    """Biên dịch (single_problem) rồi chạy với stdin cho trước."""
    with tempfile.TemporaryDirectory() as workdir:
        result = _compile_in(workdir, code, "single_problem", std, 15, msvc_compat)
        result.update(runtime_output="", runtime_error="", runtime_success=False, timed_out=False)
        if not result["compile_success"]:
            return result
        try:
            run = subprocess.run([_binary_name(workdir)], input=test_input, capture_output=True, text=True,
                                 timeout=timeout, encoding="utf-8", errors="replace")
            result["runtime_output"] = run.stdout.strip()
            result["runtime_error"] = run.stderr.strip()
            result["runtime_success"] = run.returncode == 0
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            result["runtime_error"] = "RUNTIME TIMEOUT"
        except OSError as exc:
            result["runtime_error"] = str(exc)
        return result


if __name__ == "__main__":
    test_code = """
#include <iostream>
using namespace std;
void main() {
    cout << pow(2, 10) << endl;
}
"""
    print("g++:", find_gcc())
    res = compile_cpp(test_code)
    print(f"Compile success: {res['compile_success']}\nErrors: {res['errors']}\nLog: {res['compile_log']}")
    if res["compile_success"]:
        print("Runtime output:", run_cpp_with_input(test_code)["runtime_output"])