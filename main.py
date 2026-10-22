"""
main.py -> Pipeline Chấm điểm và Phản hồi Tự động C++
Hỗ trợ cả 3 tasks của AI Challenge:
- Task 1: Rubric Grading (QWK metric chính)
- Task 2: Error Classification (Macro-F1 metric chính)
- Task 3: Pedagogical Feedback Generation (Level Compliance & Chẩn đoán)

Cách chạy:
- Chạy với LLM LoRA trên GPU (Kaggle / Local):
    python main.py --task 1 --split public_test --adapter-path ./models/lora_adapter_v3
- Chạy thử nghiệm nhanh bằng Heuristic (không cần GPU / không cần tải weights):
    python main.py --task 1 --split public_test --heuristic
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional

# Đảm bảo import được các module trong repo
REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Đảm bảo in tiếng Việt mượt mà trên console Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.data import SplitData, Submission, build_submissions, load_split
from core.schema import (
    Task1Output,
    Task1Submission,
    Task2Output,
    Task2Submission,
    Task3Output,
    Task3Submission,
    normalize_labels,
    total_score,
)
from evaluation.metrics import (
    evaluate_task1,
    evaluate_task2,
    evaluate_task3_compliance,
)
from inference.feedback_gen import FeedbackGenerator
from inference.grader import CppGrader, GradingResult


def export_submission(results: List[Dict], task: int, output_dir: str) -> str:
    """Xuất file submission theo đúng quy chuẩn BTC yêu cầu."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. File đặt theo tên task
    task_file = os.path.join(output_dir, f"task{task}_submission.json")
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 2. File predictions.json chuẩn để nộp trực tiếp lên Leaderboard
    pred_file = os.path.join(output_dir, "predictions.json")
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[Export] Đã lưu {len(results)} kết quả:")
    print(f"  -> {task_file}")
    print(f"  -> {pred_file} (File nộp Leaderboard)")
    return pred_file


def run_task1(grader: CppGrader, submissions: List[Submission]) -> List[Dict]:
    """Task 1: Chấm điểm Rubric 6 tiêu chí và tính tổng điểm qua Pydantic model."""
    print("\n" + "=" * 60)
    print("TASK 1: RUBRIC GRADING")
    print("=" * 60)

    grading_results = grader.grade_batch(submissions)

    output = []
    for r in grading_results:
        rubric_dict = r.rubric.to_dict() if hasattr(r.rubric, "to_dict") else dict(r.rubric)
        item = Task1Submission(
            sample_id=r.sample_id,
            output=Task1Output(
                rubric=rubric_dict,
                total_score=r.total_score,
            ),
        )
        output.append(item.model_dump())
    return output


def run_task2(grader: CppGrader, submissions: List[Submission],
              cached_results: Optional[List[GradingResult]] = None) -> List[Dict]:
    """Task 2: Phân loại lỗi đa nhãn (Taxonomy 10 nhãn) qua Pydantic model."""
    print("\n" + "=" * 60)
    print("TASK 2: ERROR CLASSIFICATION")
    print("=" * 60)

    grading_results = cached_results if cached_results is not None else grader.grade_batch(submissions)

    output = []
    for r in grading_results:
        item = Task2Submission(
            sample_id=r.sample_id,
            output=Task2Output(taxonomy_error=r.taxonomy_error),
        )
        output.append(item.model_dump())
    return output


def run_task3(feedback_gen: FeedbackGenerator, grader: CppGrader,
              submissions: List[Submission],
              cached_results: Optional[List[GradingResult]] = None) -> List[Dict]:
    """Task 3: Tự chẩn đoán lỗi taxonomy và sinh phản hồi sư phạm qua Pydantic model."""
    print("\n" + "=" * 60)
    print("TASK 3: PEDAGOGICAL FEEDBACK GENERATION")
    print("=" * 60)

    grading_results = cached_results if cached_results is not None else grader.grade_batch(submissions)
    error_map = {r.sample_id: r.taxonomy_error for r in grading_results}

    output = []
    n = len(submissions)
    for i, sub in enumerate(submissions):
        if (i + 1) % 10 == 0 or i == 0 or i == n - 1:
            print(f"  [Task 3] [{i+1}/{n}] Sinh feedback cho {sub.sample_id} (Level {sub.feedback_level})...")

        errors = error_map.get(sub.sample_id, [])
        feedback = feedback_gen.generate_one(sub, errors=errors, level=sub.feedback_level)

        item = Task3Submission(
            sample_id=sub.sample_id,
            output=Task3Output(
                taxonomy_error=errors,
                feedback=feedback,
            ),
        )
        output.append(item.model_dump())
    return output


def run_pipeline(args):
    """Điều phối toàn bộ quy trình."""
    print("*" * 60)
    print("C++ SLM CODE GRADING PIPELINE")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Split    : {args.split}")
    print(f"  Task     : {args.task}")
    print(f"  Mode     : {'Heuristic (No GPU)' if args.heuristic else 'LLM Model'}")
    print("*" * 60)

    # 1. Load dataset
    split_data = load_split(args.dataset, args.split)
    submissions = build_submissions(split_data)
    print(f"\nLoaded {len(submissions)} submissions từ split '{args.split}'")

    # 2. Khởi tạo LLM nếu không chạy chế độ heuristic
    llm = None
    if not args.heuristic:
        try:
            from inference.llm import LLM
            llm = LLM(
                base_model=args.model_path,
                adapter_path=args.adapter_path,
                load_in_4bit=not args.no_quant,
            )
        except Exception as e:
            print(f"\n[WARN] Không thể load LLM ({e}). Tự động fallback sang chế độ Heuristic.")
            llm = None

    # 3. Khởi tạo Grader và FeedbackGenerator
    grader = CppGrader(llm=llm, use_compiler_sandbox=not args.no_sandbox)

    train_feedback_file = str(pathlib.Path(args.dataset) / "train" / "task3_feedback.json")
    feedback_gen = FeedbackGenerator(llm=llm, train_feedback_file=train_feedback_file)

    # 4. Thực thi từng task
    results = []
    output_dir = os.path.join(args.output_dir, f"task{args.task}")

    if args.task == 1:
        results = run_task1(grader, submissions)
        # Đánh giá nội bộ nếu có ground truth
        gold_r = split_data.gold_rubrics()
        if gold_r:
            metrics = evaluate_task1(results, gold_r)
            print("\n--- Đánh giá nội bộ Task 1 ---")
            print(f"  QWK (tổng điểm) : {metrics['qwk']:.4f}")
            print(f"  MAE (tổng điểm) : {metrics['mae']:.4f}")

    elif args.task == 2:
        results = run_task2(grader, submissions)
        # Đánh giá nội bộ nếu có ground truth
        gold_t = split_data.gold_taxonomy()
        if gold_t:
            metrics = evaluate_task2(results, gold_t)
            print("\n--- Đánh giá nội bộ Task 2 ---")
            print(f"  Macro-F1 (chính): {metrics['macro_f1']:.4f}")
            print(f"  Micro-F1        : {metrics['micro_f1']:.4f}")

    elif args.task == 3:
        results = run_task3(feedback_gen, grader, submissions)
        # Đánh giá compliance
        level_map = {s.sample_id: s.feedback_level for s in submissions}
        metrics = evaluate_task3_compliance(results, level_map)
        print("\n--- Đánh giá nội bộ Task 3 ---")
        print(f"  Level Compliance: {metrics['compliance_rate']:.2%} ({metrics['compliant']}/{metrics['total']})")

    # 5. Xuất file nộp
    if results:
        export_submission(results, args.task, output_dir)

    print("\nHoàn tất pipeline thành công!")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Chấm điểm và Phản hồi Tự động C++")
    parser.add_argument("--dataset", type=str, default="./dataset",
                        help="Đường dẫn thư mục gốc dataset")
    parser.add_argument("--split", type=str, default="public_test",
                        choices=["train", "test", "public_test", "private_test"],
                        help="Split dữ liệu cần chạy")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3],
                        help="Task cần chạy: 1 (Rubric), 2 (Taxonomy), 3 (Feedback)")
    parser.add_argument("--model-path", type=str, default="unsloth/Qwen2.5-Coder-7B-Instruct",
                        help="Tên hoặc đường dẫn base model")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Đường dẫn LoRA adapter (ví dụ: ./models/lora_adapter_v3)")
    parser.add_argument("--output-dir", type=str, default="./submission",
                        help="Thư mục lưu file submission")
    parser.add_argument("--heuristic", action="store_true",
                        help="Chạy chế độ Heuristic baseline (không cần GPU / không load model)")
    parser.add_argument("--no-quant", action="store_true",
                        help="Không dùng 4-bit quantization khi load LLM")
    parser.add_argument("--no-sandbox", action="store_true",
                        help="Tắt compiler sandbox g++")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()