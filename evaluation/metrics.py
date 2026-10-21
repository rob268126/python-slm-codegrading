"""
evaluation/metrics.py
Tính toán các metric đánh giá theo đúng quy định của BTC (DoAn-LLM-Challenge.docx / task.md):
- Task 1: QWK trên total_score (metric chính), MAE, exact-match từng chiều rubric.
- Task 2: Macro-F1 (metric chính), Micro-F1 trên 10 nhãn taxonomy.
- Task 3: Level Compliance Rate (Level 1/2 cấm code/lời giải).
"""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, f1_score

from core.schema import (
    RUBRIC_DIMS,
    TAXONOMY_LABELS,
    clamp_rubric,
    normalize_labels,
    total_score,
)


def quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """QWK trên tổng điểm (0-10) — metric xếp hạng chính của Task 1."""
    if len(y_true) == 0:
        return 0.0
    try:
        score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        return float(score) if not np.isnan(score) else 0.0
    except Exception:
        return 0.0


def evaluate_task1(
    predictions: Sequence[Mapping],
    gold_rubrics: Mapping[str, Mapping[str, int]],
) -> Dict[str, float]:
    """Đánh giá Task 1: QWK, MAE, và exact match theo từng chiều rubric."""
    y_true_total: List[int] = []
    y_pred_total: List[int] = []
    dim_matches: Dict[str, int] = {d: 0 for d in RUBRIC_DIMS}
    n = 0

    for pred in predictions:
        sid = pred.get("sample_id", "")
        if sid not in gold_rubrics:
            continue
        gt_r = clamp_rubric(gold_rubrics[sid])
        pred_out = pred.get("output", {})
        pred_r = clamp_rubric(pred_out.get("rubric", {}))

        gt_tot = total_score(gt_r)
        pred_tot = pred_out.get("total_score", total_score(pred_r))

        y_true_total.append(gt_tot)
        y_pred_total.append(int(pred_tot))
        n += 1

        for d in RUBRIC_DIMS:
            if pred_r[d] == gt_r[d]:
                dim_matches[d] += 1

    if n == 0:
        return {"qwk": 0.0, "mae": 0.0, "samples": 0}

    results: Dict[str, float] = {
        "qwk": quadratic_weighted_kappa(y_true_total, y_pred_total),
        "mae": float(mean_absolute_error(y_true_total, y_pred_total)),
        "samples": float(n),
    }
    for d in RUBRIC_DIMS:
        results[f"exact_{d}"] = dim_matches[d] / n
    return results


def evaluate_task2(
    predictions: Sequence[Mapping],
    gold_taxonomy: Mapping[str, Sequence[str]],
) -> Dict[str, float]:
    """Đánh giá Task 2: Macro-F1 (metric chính) và Micro-F1 trên 10 nhãn taxonomy."""
    label_to_idx = {label: i for i, label in enumerate(TAXONOMY_LABELS)}
    n_labels = len(TAXONOMY_LABELS)

    y_true: List[List[int]] = []
    y_pred: List[List[int]] = []

    for pred in predictions:
        sid = pred.get("sample_id", "")
        if sid not in gold_taxonomy:
            continue
        gt_labels = normalize_labels(gold_taxonomy[sid])
        pred_labels = normalize_labels(pred.get("output", {}).get("taxonomy_error", []))

        t_vec = [0] * n_labels
        for l in gt_labels:
            if l in label_to_idx:
                t_vec[label_to_idx[l]] = 1

        p_vec = [0] * n_labels
        for l in pred_labels:
            if l in label_to_idx:
                p_vec[label_to_idx[l]] = 1

        y_true.append(t_vec)
        y_pred.append(p_vec)

    if not y_true:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "samples": 0}

    y_true_mat = np.array(y_true)
    y_pred_mat = np.array(y_pred)

    macro_f1 = float(f1_score(y_true_mat, y_pred_mat, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true_mat, y_pred_mat, average="micro", zero_division=0))

    results: Dict[str, float] = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "samples": float(len(y_true)),
    }
    # F1 từng nhãn
    per_label_f1 = f1_score(y_true_mat, y_pred_mat, average=None, zero_division=0)
    for i, label in enumerate(TAXONOMY_LABELS):
        results[f"f1_{label}"] = float(per_label_f1[i])

    return results


# Kiểm tra vi phạm Level 1 và 2 (cấm đưa code sửa / giải pháp hoàn chỉnh)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_CODE_LINE_RE = re.compile(r"^\s*(#include|int\s+main|void\s+main|using\s+namespace|cin\s*>>|cout\s*<<)", re.MULTILINE)
_SOLUTION_PATTERNS = re.compile(r"(lời giải mẫu|đáp án hoàn chỉnh|code hoàn chỉnh|toàn bộ code|full code)", re.IGNORECASE)


def is_level_compliant(feedback: str, level: int) -> bool:
    """Trả về True nếu feedback tuân thủ đúng mức kiểm soát (Level 1-4)."""
    if level >= 3:
        return True  # Level 3 cho phép snippet nhỏ, Level 4 cho phép code hoàn chỉnh
    # Level 1 và 2: CẤM tuyệt đối code block và lời giải mẫu
    if _CODE_BLOCK_RE.search(feedback):
        return False
    if _CODE_LINE_RE.search(feedback):
        return False
    if _SOLUTION_PATTERNS.search(feedback):
        return False
    return True


def evaluate_task3_compliance(
    predictions: Sequence[Mapping],
    levels: Optional[Mapping[str, int]] = None,
) -> Dict[str, float]:
    """Đánh giá tỷ lệ tuân thủ Level Compliance cho Task 3."""
    total = 0
    compliant = 0
    for pred in predictions:
        sid = pred.get("sample_id", "")
        feedback = pred.get("output", {}).get("feedback", "")
        # Lấy level từ levels map nếu có, mặc định là Level 2
        level = levels.get(sid, 2) if levels else 2
        total += 1
        if is_level_compliant(feedback, level):
            compliant += 1

    rate = (compliant / total) if total > 0 else 0.0
    return {"compliance_rate": rate, "compliant": compliant, "total": total}