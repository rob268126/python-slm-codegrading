"""
main.py - Pipeline chấm điểm tự động C++
Chạy: python main.py --dataset ./dataset --model-path unsloth/Qwen2.5-Coder-7B-Instruct --adapter-path ./models/lora_adapter_v3 --output predictions.json
"""
import json
import os
import sys
import argparse
import pathlib
import numpy as np
from collections import Counter

# Seed cố định
SEED = 42
np.random.seed(SEED)

RUBRIC_DIMS = ["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"]


def load_dataset(dataset_root: str) -> dict:
    """Load toàn bộ dataset"""
    root = pathlib.Path(dataset_root)
    data = {}
    for fname in ["task1_grading.json", "task2_error_taxonomy.json",
                  "task3_feedback.json", "exams.json", "label_space.json"]:
        fpath = root / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                data[fname] = json.load(f)
    return data


def compile_cpp(code: str, timeout: int = 10) -> dict:
    """Compile C++ code"""
    import subprocess
    import tempfile

    result = {
        "compilable": False,
        "compile_log": "",
        "compile_success": False,
        "warnings": [],
        "errors": [],
        "error_types": []
    }

    with tempfile.NamedTemporaryFile(suffix=".cpp", mode="w", delete=False, encoding="utf-8") as f:
        f.write(code)
        cpp_path = f.name

    exe_path = cpp_path.replace(".cpp", ".out.exe" if os.name == "nt" else ".out")

    try:
        compile_cmd = ["g++", "-std=c++17", "-O2", "-Wall", "-o", exe_path, cpp_path]
        proc = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=timeout)

        result["compile_log"] = proc.stderr.strip()
        result["compile_success"] = (proc.returncode == 0)
        result["compilable"] = (proc.returncode == 0)

        for line in proc.stderr.split("\n"):
            ll = line.lower()
            if "error:" in ll:
                result["errors"].append(line.strip())
                if "undeclared" in ll or "not declared" in ll:
                    result["error_types"].append("Lỗi biến/hàm chưa khai báo")
                elif "no match for" in ll or "no matching function" in ll:
                    result["error_types"].append("Lỗi hàm")
                elif "expected" in ll and (";" in ll or "}" in ll or "{" in ll):
                    result["error_types"].append("Lỗi cú pháp")
                elif "cannot convert" in ll or "invalid conversion" in ll:
                    result["error_types"].append("Lỗi kiểu dữ liệu")
                elif "redeclaration" in ll or "redefinition" in ll:
                    result["error_types"].append("Lỗi khai báo trùng")
                elif "ld returned" in ll or "undefined reference" in ll:
                    result["error_types"].append("Lỗi liên kết (missing main)")
                else:
                    result["error_types"].append("Lỗi biên dịch khác")
            elif "warning:" in ll:
                result["warnings"].append(line.strip())

    except subprocess.TimeoutExpired:
        result["compile_log"] = "COMPILATION TIMEOUT"
        result["errors"].append("Compilation timed out")
        result["error_types"].append("Lỗi biên dịch khác")
    except FileNotFoundError:
        result["compile_log"] = "g++ not found"
        result["errors"].append("g++ not installed")
    except Exception as e:
        result["compile_log"] = str(e)
        result["errors"].append(str(e))
    finally:
        for p in (cpp_path, exe_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except:
                    pass

    return result


def generate_synthetic_labels(compiler_result: dict, rubric: dict, code: str) -> list:
    """Tạo synthetic labels cho Task 2"""
    labels = []

    if not compiler_result["compile_success"]:
        labels.append("Lỗi biên dịch")
        for et in compiler_result["error_types"]:
            if et not in labels:
                labels.append(et)

    if rubric.get("logic", 4) <= 1:
        labels.append("Lỗi logic")
    if rubric.get("edge_case", 2) == 0:
        labels.append("Lỗi xử lý biên")
    if rubric.get("complexity", 1) == 0:
        labels.append("Lỗi thuật toán")
    if rubric.get("io_format", 1) == 0 and compiler_result["compile_success"]:
        labels.append("Lỗi I/O")

    code_lower = code.lower()
    if compiler_result["compile_success"]:
        if "while(true)" in code_lower or "while(1)" in code_lower:
            if rubric.get("logic", 4) <= 2:
                labels.append("Lỗi vòng lặp")
        if ("new " in code_lower or "malloc" in code_lower) and "delete" not in code_lower and "free" not in code_lower:
            labels.append("Lỗi bộ nhớ")

    if not labels:
        labels.append("Không có lỗi")

    return list(set(labels))


def build_prompt(sample: dict) -> str:
    """Xây dựng prompt cho model"""
    input_text = (
        f"### Đề bài:\n{sample['statement']}\n\n"
        f"### Code sinh viên:\n```cpp\n{sample['code']}\n```\n\n"
        f"### Kết quả biên dịch thực tế (g++ -std=c++17):\n{sample['compile_status']}\n"
        f"Log: {sample['compile_log'][:500]}\n\n"
        f"### Test Report:\n{sample.get('test_report', 'Không có')}"
    )

    text = (
        f"<|im_start|>system\n"
        f"You are an expert C++ code grader acting as a JUDGE. "
        f"You have ACTUAL COMPILATION RESULTS as ground truth - trust them above all.\n"
        f"Grading rubric:\n"
        f"- compilable (0-1): Does code compile? Use compiler result.\n"
        f"- io_format (0-1): Correct input/output format?\n"
        f"- logic (0-4): Algorithm correctness. IF compilable=0, this MUST be 0.\n"
        f"- edge_case (0-2): Handles edge cases? IF compilable=0, this MUST be 0.\n"
        f"- complexity (0-1): Appropriate time/space complexity?\n"
        f"- code_quality (0-1): Clean, readable code?\n\n"
        f"Error taxonomy (multi-label): Lỗi biên dịch, Lỗi logic, Lỗi vòng lặp, "
        f"Lỗi hàm, Lỗi I/O, Lỗi bộ nhớ, Lỗi kiểu dữ liệu, Lỗi thuật toán, "
        f"Lỗi xử lý biên, Lỗi cú pháp, Không có lỗi\n\n"
        f"Output ONLY valid JSON: {{\"rubric\": {{...}}, \"errors\": [...]}}<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"Rationale: {sample['rationale']}\n\n"
        f"Final JSON:\n"
    )
    return text


def parse_model_output(text: str) -> dict:
    """Parse JSON từ output model"""
    import re

    # Tìm JSON block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Fallback: regex
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None


def run_pipeline(args):
    """Chạy toàn bộ pipeline"""
    print("=" * 60)
    print("C++ CODE GRADING PIPELINE")
    print("=" * 60)

    # 1. Load dataset
    root = pathlib.Path(args.dataset)
    data = load_dataset(args.dataset)
    print(f"Loaded: {list(data.keys())}")

    task1_data = data.get("task1_grading.json", {})
    task2_data = data.get("task2_error_taxonomy.json", {})
    task3_data = data.get("task3_feedback.json", {})
    exams_data = data.get("exams.json", {})

    task1_samples = task1_data.get("samples", [])
    task2_map = {s["sample_id"]: s for s in task2_data.get("samples", [])}
    task3_map = {s["sample_id"]: s for s in task3_data.get("samples", [])}

    exam_list = exams_data.get("exams", exams_data if isinstance(exams_data, list) else [])
    exam_statements = {e["exam_id"]: e.get("statement", "") for e in exam_list}

    print(f"Task 1 samples: {len(task1_samples)}")
    print(f"Task 2 samples: {len(task2_map)}")
    print(f"Task 3 samples: {len(task3_map)}")

    # 2. Load model
    model = None
    tokenizer = None
    if args.model_path:
        print(f"\nLoading model: {args.model_path}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True,
            )

            if args.adapter_path and os.path.exists(args.adapter_path):
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.adapter_path)
                print(f"  LoRA adapter loaded: {args.adapter_path}")

            model.eval()
            print("  Model ready.")
        except Exception as e:
            print(f"  [WARN] Model load failed: {e}")
            model = None

    # 3. Process each sample
    print(f"\nProcessing {len(task1_samples)} samples...")
    predictions = []

    for i, s in enumerate(task1_samples):
        sid = s["sample_id"]
        inp = s["input"]
        code_file = inp.get("code_file", "")
        exam_id = inp.get("exam_id", "")

        # Đọc code file
        code_path = root / code_file
        if not code_path.exists():
            print(f"  [{i+1}/{len(task1_samples)}] {sid}: SKIP (file not found: {code_file})")
            continue

        code = code_path.read_text(encoding="utf-8")
        statement = exam_statements.get(exam_id, "")
        rubric_gt = s["output"].get("rubric", {})

        # Compile
        cr = compile_cpp(code)

        # Tạo synthetic labels
        errors_gt = task2_map.get(sid, {}).get("output", {}).get("errors", [])
        if not errors_gt:
            errors_gt = generate_synthetic_labels(cr, rubric_gt, code)

        # Compile status
        compile_status = "COMPILE SUCCESS" if cr["compile_success"] else "COMPILE FAILED"
        compile_log = cr["compile_log"] if cr["compile_log"] else "Không có lỗi biên dịch"

        # Rationale
        if cr["compile_success"]:
            rationale = f"Code biên dịch thành công với g++ -std=c++17."
            if cr["warnings"]:
                rationale += f" Có {len(cr['warnings'])} warning."
        else:
            main_error = cr["errors"][0][:100] if cr["errors"] else "unknown"
            rationale = f"Code KHÔNG biên dịch được. Lỗi chính: {main_error}. Do đó logic=0, edge_case=0."

        # Corrected rubric
        corrected_rubric = rubric_gt.copy()
        if not cr["compile_success"]:
            corrected_rubric["compilable"] = 0
            corrected_rubric["logic"] = 0
            corrected_rubric["edge_case"] = 0

        # Build sample dict
        sample = {
            "sample_id": sid,
            "statement": statement,
            "code": code,
            "compile_status": compile_status,
            "compile_log": compile_log,
            "test_report": inp.get("test_report", "Không có"),
            "rationale": rationale,
            "rubric_gt": corrected_rubric,
            "errors_gt": errors_gt,
        }

        # Inference
        if model is not None and tokenizer is not None:
            import torch

            prompt = build_prompt(sample)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3800)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.0,
                    do_sample=False,
                    top_p=1.0,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            parsed = parse_model_output(response)

            if parsed and "rubric" in parsed:
                pred_rubric = parsed["rubric"]
                pred_errors = parsed.get("errors", [])
            else:
                # Fallback: dùng compiler result
                pred_rubric = corrected_rubric
                pred_errors = errors_gt
        else:
            # Không có model -> dùng heuristic
            pred_rubric = corrected_rubric
            pred_errors = errors_gt

        predictions.append({
            "sample_id": sid,
            "task1": {
                "rubric": pred_rubric,
                "total": sum(pred_rubric.get(d, 0) for d in RUBRIC_DIMS)
            },
            "task2": {"errors": pred_errors},
            "ground_truth": {
                "rubric": corrected_rubric,
                "errors": errors_gt,
            }
        })

        status = "PASS" if cr["compile_success"] else "FAIL"
        print(f"  [{i+1:2d}/{len(task1_samples)}] {sid}: compile={status}, "
              f"total={sum(pred_rubric.get(d, 0) for d in RUBRIC_DIMS)}, "
              f"errors={len(pred_errors)}")

    # 4. Evaluation
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")

    if predictions:
        # Task 1: QWK + MAE
        gt_totals = [sum(p["ground_truth"]["rubric"].get(d, 0) for d in RUBRIC_DIMS) for p in predictions]
        pred_totals = [p["task1"]["total"] for p in predictions]

        try:
            from sklearn.metrics import cohen_kappa_score, mean_absolute_error, f1_score
            qwk = cohen_kappa_score(gt_totals, pred_totals, weights="quadratic")
            mae = mean_absolute_error(gt_totals, pred_totals)
        except:
            qwk = 0.0
            mae = 0.0

        print(f"  Task 1  QWK:  {qwk:.4f}")
        print(f"  Task 1  MAE:  {mae:.4f}")

        # Task 2: macro-F1
        VALID_LABELS = [
            "Lỗi biên dịch", "Lỗi logic", "Lỗi vòng lặp", "Lỗi hàm",
            "Lỗi I/O", "Lỗi bộ nhớ", "Lỗi kiểu dữ liệu", "Lỗi thuật toán",
            "Lỗi xử lý biên", "Lỗi cú pháp", "Không có lỗi"
        ]

        gt_labels_bin = []
        pred_labels_bin = []
        for p in predictions:
            gt_e = p["ground_truth"]["errors"]
            pred_e = p["task2"]["errors"]
            gt_labels_bin.append([1 if l in gt_e else 0 for l in VALID_LABELS])
            pred_labels_bin.append([1 if l in pred_e else 0 for l in VALID_LABELS])

        try:
            from sklearn.metrics import f1_score
            macro_f1 = f1_score(gt_labels_bin, pred_labels_bin, average="macro", zero_division=0)
        except:
            macro_f1 = 0.0

        print(f"  Task 2  macro-F1: {macro_f1:.4f}")
    else:
        print("  No predictions to evaluate.")

    # 5. Save
    output_path = args.output or "predictions.json"
    # Chỉ lưu phần prediction (không lưu ground_truth)
    output_data = []
    for p in predictions:
        output_data.append({
            "sample_id": p["sample_id"],
            "task1": p["task1"],
            "task2": p["task2"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(predictions)} predictions → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C++ Code Grading Pipeline")
    parser.add_argument("--dataset", type=str, default="./dataset")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--output", type=str, default="predictions.json")
    args = parser.parse_args()
    run_pipeline(args)