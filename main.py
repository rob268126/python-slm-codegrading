"""
main.py -> Pipeline chấm điểm tự động C++
Chạy: python main.py --dataset ./dataset --task 1 --split public_test
"""
import json
import os
import sys
import argparse
import pathlib
import torch
from typing import List, Dict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from inference.grader import CppGrader, GradingResult
from inference.feedback_gen import FeedbackGenerator
from sandbox.cpp_compiler import compile_cpp
from evaluation.metrics import evaluate_all

VALID_LABELS = {
    "Lỗi biên dịch", "Lỗi logic", "Lỗi vòng lặp", "Lỗi hàm", "Lỗi I/O",
    "Lỗi bộ nhớ", "Lỗi kiểu dữ liệu", "Lỗi thuật toán", "Lỗi xử lý biên", "Lỗi cú pháp"
}

LABEL_MAP = {
    "Lỗi biến/hàm chưa khai báo": "Lỗi cú pháp",
    "Lỗi liên kết (missing main)": "Lỗi biên dịch",
    "Lỗi biên dịch khác": "Lỗi biên dịch",
    "Không có lỗi": None,
    "Lỗi syntax": "Lỗi cú pháp",
    "Lỗi logic sai": "Lỗi logic",
}

def sanitize_taxonomy_errors(errors_list):
    """Lọc và map các nhãn lỗi về đúng 10 nhãn chính thức."""
    if not isinstance(errors_list, list):
        return []
    
    valid = []
    for e in errors_list:
        if not isinstance(e, str):
            continue
        e = e.strip()
        
        if e in VALID_LABELS:
            if e not in valid:
                valid.append(e)
        elif e in LABEL_MAP and LABEL_MAP[e] is not None:
            mapped = LABEL_MAP[e]
            if mapped not in valid:
                valid.append(mapped)
    
    return valid

def load_dataset(dataset_root: str, split: str = "train") -> dict:
    """Load dataset từ thư mục train/test/public_test/private_test"""
    root = pathlib.Path(dataset_root) / split
    
    if not root.exists():
        root = pathlib.Path(dataset_root)
    
    data = {}
    for fname in ["task1_grading.json", "task2_error_taxonomy.json",
                  "task3_feedback.json", "task3_requested_levels.json",
                  "exams.json", "label_space.json"]:
        fpath = root / fname
        if fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                data[fname] = json.load(f)
    
    # Load submissions
    submissions_dir = root / "submissions"
    if submissions_dir.exists():
        data["submissions_dir"] = submissions_dir
    
    return data


def export_submission(results: List[Dict], task: int, output_dir: str):
    """Xuất submission JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    submission_data = results
    
    output_file = f"task{task}_submission.json"
    output_path = os.path.join(output_dir, output_file)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(submission_data)} predictions → {output_path}")
    return output_path


def build_submissions(data: dict) -> List[Dict]:
    """Build danh sách submissions dùng chung cho Task 1, 2, 3"""
    task1_data = data.get("task1_grading.json", {})
    samples = task1_data.get("samples", [])
    exams_data = data.get("exams.json", {})
    
    submissions = []
    for s in samples:
        inp = s["input"]
        code_file = inp.get("code_file", "")
        
        # Xử lý pat
        if "submissions_dir" in data:
            code_path = data["submissions_dir"].parent / code_file
            if not code_path.exists():
                code_path = data["submissions_dir"] / pathlib.Path(code_file).name
        else:
            code_path = pathlib.Path(code_file)
        
        if not code_path.exists():
            print(f"  File not found: {code_file}")
            continue
        
        code = code_path.read_text(encoding="utf-8")
        exam_id = inp.get("exam_id", "")
        
        # Lấy statement từ exams.json
        statement = ""
        for exam in exams_data.get("exams", []):
            if exam.get("exam_id") == exam_id:
                statement = exam.get("statement", "")
                break
        
        submissions.append({
            "sample_id": s["sample_id"],
            "exam_id": exam_id,
            "statement": statement,
            "code": code,
            "compile_log": inp.get("compile_log", ""),
            "test_report": inp.get("test_report", "")
        })
    
    return submissions


def process_task1(grader: CppGrader, data: dict, split: str) -> List[Dict]:
    """Task 1: Rubric Grading - tự tính lại total_score"""
    print("\n" + "#"*60)
    print(f"TASK 1: RUBRIC GRADING ({split})")
    
    submissions = build_submissions(data)
    
    results = grader.grade_batch(submissions)
    results = grader.apply_prerequisite_policy(results, submissions)

    output = []
    for r in results:
        rubric = r.rubric
        total_score = sum(
            rubric.get(d, 0) 
            for d in ["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"]
        )
        
        output.append({
            "sample_id": r.sample_id,
            "output": {
                "rubric": rubric,
                "total_score": total_score
            }
        })
    
    return output


def process_task2(grader: CppGrader, data: dict, split: str) -> List[Dict]:
    """Task 2: Error Classification"""
    print("\n" + "#"*60)
    print(f"TASK 2: ERROR CLASSIFICATION ({split})")
    
    submissions = build_submissions(data)
    
    results = grader.grade_batch(submissions)
    results = grader.apply_prerequisite_policy(results, submissions)

    output = []
    for r in results:
        cleaned_errors = sanitize_taxonomy_errors(r.errors)
        
        output.append({
            "sample_id": r.sample_id,
            "output": {
                "taxonomy_error": cleaned_errors  # Dùng cleaned_errors thay vì r.errors
            }
        })
    
    return output


def process_task3(feedback_gen: FeedbackGenerator, grader: CppGrader, 
                  data: dict, split: str) -> List[Dict]:
    """Task 3: Feedback Generation + RAG 1-Shot"""
    print("\n" + "#"*60)
    print(f"TASK 3: FEEDBACK GENERATION ({split})")
    
    # Load requested levels
    task3_levels = data.get("task3_requested_levels.json", {})
    if isinstance(task3_levels, list):
        level_map = {item["sample_id"]: item.get("requested_level", 2) for item in task3_levels}
    else:
        level_map = {}
    
    submissions = build_submissions(data)
    
    # Lấy errors từ grader
    results = grader.grade_batch(submissions)
    results = grader.apply_prerequisite_policy(results, submissions)
    
    # Map sample_id -> errors
    error_map = {r.sample_id: r.errors for r in results}
    
    output = []
    for i, sub in enumerate(submissions):
        sid = sub["sample_id"]
        level = level_map.get(sid, 2)
        errors = error_map.get(sid, [])
        
        print(f"  [{i+1}/{len(submissions)}] {sid} (Level {level})")
        
        result = feedback_gen.generate(
            code=sub["code"],
            statement=sub["statement"],
            errors=errors,
            level=level
        )

        cleaned_errors = sanitize_taxonomy_errors(errors)
        
        output.append({
            "sample_id": sid,
            "output": {
                "taxonomy_error": cleaned_errors,  # Dùng cleaned_errors thay vì errors
                "feedback": result["feedback"]
            }
        })
    
    return output

def run_pipeline(args):
    """Chạy pipeline"""
    print("="*60)
    print("C++ CODE GRADING PIPELINE")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Task: {args.task}")
    
    # 1. Load dataset
    data = load_dataset(args.dataset, args.split)
    print(f"Loaded dataset from {args.dataset}/{args.split}")
    
    # 2. Load model
    grader = CppGrader(
        base_model=args.model_path,
        adapter_path=args.adapter_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 3. Process theo task
    results = []
    if args.task == 1:
        results = process_task1(grader, data, args.split)
        output_dir = os.path.join(args.output_dir, "task1")
    elif args.task == 2:
        results = process_task2(grader, data, args.split)
        output_dir = os.path.join(args.output_dir, "task2")
    elif args.task == 3:
        # TODO: Load train_examples cho RAG
        # train_examples = load_task3_train_examples(...)
        feedback_gen = FeedbackGenerator(
            model=grader.model,
            tokenizer=grader.tokenizer,
            # train_examples=train_examples   # bổ sung synthesis data
        )
        results = process_task3(feedback_gen, grader, data, args.split)
        output_dir = os.path.join(args.output_dir, "task3")
    
    # 4. Export submission
    if results:
        export_submission(results, args.task, output_dir)
    
    print("\nDone")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C++ Code Grading Pipeline")
    parser.add_argument("--dataset", type=str, default="./dataset",
                        help="Path to dataset root directory")
    parser.add_argument("--model-path", type=str, 
                        default="unsloth/Qwen2.5-Coder-7B-Instruct",
                        help="Base model path or HuggingFace repo")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3],
                        help="Task to run: 1 (grading), 2 (taxonomy), 3 (feedback)")
    parser.add_argument("--split", type=str, default="public_test",
                        choices=["train", "test", "public_test", "private_test"],
                        help="Dataset split to process")
    parser.add_argument("--output-dir", type=str, default="./submission",
                        help="Directory to save submission files")
    args = parser.parse_args()
    run_pipeline(args)