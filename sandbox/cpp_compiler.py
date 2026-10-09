"""
C++ Compiler Sandbox - Ground Truth từ kết quả biên dịch thực tế.
Hỗ trợ Windows với fallback path tìm g++.
"""
import subprocess
import tempfile
import os
import sys
from typing import Optional

# environment g++ check 
def find_gcc():
    """Tìm đường dẫn g++ trên hệ thống."""
    # PATH 
    try:
        result = subprocess.run(["g++", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return "g++"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # try path 
    possible_paths = [
        r"C:\msys64\ucrt64\bin\g++.exe",
        r"C:\msys64\mingw64\bin\g++.exe",
        r"C:\MinGW\bin\g++.exe",
        r"C:\TDM-GCC-64\bin\g++.exe",
        r"C:\Program Files\mingw64\bin\g++.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def compile_cpp(code: str, timeout: int = 15) -> dict:
    """
    Compile C++ code, trả về kết quả chi tiết.
    
    Returns:
        dict với keys:
        - compilable: bool
        - compile_success: bool
        - compile_log: str
        - errors: list[str]
        - warnings: list[str]
        - error_types: list[str]
    """
    result = {
        "compilable": False,
        "compile_success": False,
        "compile_log": "",
        "errors": [],
        "warnings": [],
        "error_types": [],
    }

    # use g++
    gpp_path = find_gcc()
    if gpp_path is None:
        result["compile_log"] = "g++ not found. Please install MinGW/g++ and add to PATH."
        result["errors"].append("g++ not installed")
        result["error_types"].append("Lỗi môi trường")
        return result

    # file tạm
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".cpp", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            cpp_path = f.name
    except Exception as e:
        result["compile_log"] = f"Cannot create temp file: {str(e)}"
        result["errors"].append(f"Temp file error: {str(e)}")
        return result

    # Đường dẫn exe
    exe_path = cpp_path.replace(".cpp", ".exe")

    try:
        # Compile command
        compile_cmd = [
            gpp_path,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-o", exe_path,
            cpp_path
        ]
        
        proc = subprocess.run(
            compile_cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )

        result["compile_log"] = proc.stderr.strip()
        result["compile_success"] = proc.returncode == 0
        result["compilable"] = proc.returncode == 0

        # Parse errors + warnings
        if proc.stderr:
            for line in proc.stderr.split("\n"):
                ll = line.lower().strip()
                if not ll:
                    continue
                    
                if "error:" in ll or "error" in ll:
                    result["errors"].append(line.strip())
                    # Phân loại lỗi
                    if "undeclared" in ll or "not declared" in ll:
                        result["error_types"].append("Lỗi biến/hàm chưa khai báo")
                    elif "no match for" in ll or "no matching function" in ll:
                        result["error_types"].append("Lỗi hàm")
                    elif "expected" in ll and any(c in ll for c in (";", "}", "{", ")")):
                        result["error_types"].append("Lỗi cú pháp")
                    elif "cannot convert" in ll or "invalid conversion" in ll:
                        result["error_types"].append("Lỗi kiểu dữ liệu")
                    elif "redeclaration" in ll or "redefinition" in ll:
                        result["error_types"].append("Lỗi khai báo trùng")
                    elif "undefined reference" in ll or "ld returned" in ll:
                        result["error_types"].append("Lỗi liên kết (missing main)")
                    elif "no such file" in ll:
                        result["error_types"].append("Lỗi thiếu header file")
                    else:
                        result["error_types"].append("Lỗi biên dịch khác")
                elif "warning:" in ll:
                    result["warnings"].append(line.strip())

    except subprocess.TimeoutExpired:
        result["compile_log"] = "COMPILATION TIMEOUT"
        result["errors"].append("Compilation timed out")
        result["error_types"].append("Lỗi timeout")
    except FileNotFoundError:
        result["compile_log"] = "g++ not found. Please install MinGW/g++."
        result["errors"].append("g++ not installed")
        result["error_types"].append("Lỗi môi trường")
    except Exception as e:
        result["compile_log"] = f"Unexpected error: {str(e)}"
        result["errors"].append(str(e))
        result["error_types"].append("Lỗi hệ thống")
    finally:
        # Dọn dẹp file tạm
        for p in (cpp_path, exe_path):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass

    return result


def run_cpp_with_input(code: str, test_input: str = "", timeout: int = 5) -> dict:
    """Compile VÀ chạy code với input cho trước."""
    result = compile_cpp(code)
    result["runtime_output"] = ""
    result["runtime_error"] = ""
    result["runtime_success"] = False
    result["timed_out"] = False

    if not result["compile_success"]:
        return result

    # Tìm g++
    gpp_path = find_gcc()
    if gpp_path is None:
        return result

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".cpp", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            cpp_path = f.name

        exe_path = cpp_path.replace(".cpp", ".exe")

        # recompile
        subprocess.run(
            [gpp_path, "-std=c++17", "-O2", "-o", exe_path, cpp_path],
            capture_output=True,
            timeout=15,
        )
        
        # Chạy input
        run_proc = subprocess.run(
            [exe_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )
        
        result["runtime_output"] = run_proc.stdout.strip()
        result["runtime_error"] = run_proc.stderr.strip()
        result["runtime_success"] = run_proc.returncode == 0

    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["runtime_error"] = "RUNTIME TIMEOUT"
    except Exception as e:
        result["runtime_error"] = str(e)
    finally:
        for p in (cpp_path, exe_path):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass

    return result

if __name__ == "__main__":
    # Test code
    test_code = """
#include <iostream>
using namespace std;
int main() {
    cout << "Hello World" << endl;
    return 0;
}
"""
    print("Testing C++ compiler...")
    result = compile_cpp(test_code)
    print(f"Compile success: {result['compile_success']}")
    print(f"Errors: {result['errors']}")
    print(f"Log: {result['compile_log']}")
    
    if result["compile_success"]:
        print("\nRunning with input...")
        run_result = run_cpp_with_input(test_code)
        print(f"Runtime output: {run_result['runtime_output']}")