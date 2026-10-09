"""
evaluation/metrics.py
Tính QWK, MAE, exact-match (Task 1), macro-F1 (Task 2), Level compliance (Task 3)
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, f1_score
from typing import List, Dict


def quadratic_weighted_kappa(y_true, y_pred):
    """QWK - metric chính Task 1."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def evaluate_all(predictions: list, task1: dict, task2: dict, task3: dict):
    """Đánh giá"""
    
    # Task 1: QWK + MAE 
    print("\n--- Task 1: Rubric Grading ---")
    gt_map = {s["sample_id"]: s["output"]["rubric"] for s in task1["samples"]}
    
    totals_true, totals_pred = [], []
    for pred in predictions:
        sid = pred["sample_id"]
        if sid in gt_map:
            gt = gt_map[sid]
            p = pred["task1"]["rubric"]
            totals_true.append(sum(gt.values()))
            totals_pred.append(sum(p.values()))
    
    if totals_true:
        qwk = quadratic_weighted_kappa(totals_true, totals_pred)
        mae = mean_absolute_error(totals_true, totals_pred)
        print(f"  QWK (tổng điểm): {qwk:.4f}")
        print(f"  MAE (tổng điểm): {mae:.4f}")
    else:
        print("Không có ground truth để evaluate Task 1")

    # Task 2: Macro-F1
    print("\n--- Task 2: Error Classification ---")
    gt_errors_map = {s["sample_id"]: s["output"].get("errors", []) for s in task2["samples"]}
    
    all_empty = all(len(v) == 0 for v in gt_errors_map.values())
    if all_empty:
        print("  Tất cả ground truth errors đều RỖNG - không thể tính macro-F1")
        print("  -> Cần tạo synthetic labels từ compiler output")
    else:
        # macro-F1
        pass

    # Task 3: Level Compliance
    print("\n--- Task 3: Feedback Generation ---")
    compliant_count = 0
    total_feedback = 0
    for pred in predictions:
        if "task3" in pred:
            total_feedback += 1
            if pred["task3"].get("level_compliant", False):
                compliant_count += 1
    
    if total_feedback > 0:
        compliance_rate = compliant_count / total_feedback
        print(f"  Level Compliance Rate: {compliance_rate:.2%} ({compliant_count}/{total_feedback})")
    else:
        print("Chưa có feedback nào được sinh")