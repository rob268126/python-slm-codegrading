"""
inference/ensemble.py
Chiến lược Ensemble kết hợp nhiều Judges (ví dụ Qwen LoRA + Rule-based / JEV).
Đảm bảo output luôn tuân thủ strict rubric clamping và taxonomy normalization.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence

from core.schema import (
    RUBRIC_DIMS,
    RUBRIC_RANGES,
    clamp_rubric,
    normalize_labels,
    total_score,
)


class GradingEnsemble:
    """Kết hợp kết quả từ nhiều judges:
    - Rubric: Weighted average rồi làm tròn. Đối với 'compilable', áp dụng nguyên tắc conservative (min).
    - Errors: Majority voting qua threshold, chuẩn hóa qua `normalize_labels`.
    """

    def __init__(self, error_threshold: float = 0.5):
        self.error_threshold = error_threshold

    def ensemble_rubric(
        self,
        predictions: Sequence[Mapping],
        weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, int]:
        if not predictions:
            return {d: 0 for d in RUBRIC_DIMS}

        n = len(predictions)
        if weights is None or len(weights) != n:
            weights = [1.0 / n] * n

        total_w = sum(weights) or 1.0
        norm_weights = [w / total_w for w in weights]

        result: Dict[str, int] = {}
        for dim in RUBRIC_DIMS:
            lo, hi = RUBRIC_RANGES[dim]
            if dim == "compilable":
                # Conservative: nếu có bất kỳ judge nào nói compilable=0 -> 0
                vals = [p.get("rubric", {}).get(dim, 1) for p in predictions]
                result[dim] = min(vals) if vals else 0
            else:
                weighted_val = sum(
                    w * p.get("rubric", {}).get(dim, 0)
                    for w, p in zip(norm_weights, predictions)
                )
                result[dim] = max(lo, min(hi, int(round(weighted_val))))

        return clamp_rubric(result)

    def ensemble_errors(
        self,
        predictions: Sequence[Mapping],
        threshold: Optional[float] = None,
    ) -> List[str]:
        if threshold is None:
            threshold = self.error_threshold

        if not predictions:
            return []

        counts = Counter()
        for p in predictions:
            raw = p.get("taxonomy_error") or p.get("errors") or []
            for e in normalize_labels(raw):
                counts[e] += 1

        min_votes = max(1, int(threshold * len(predictions)))
        voted = [err for err, cnt in counts.items() if cnt >= min_votes]
        return normalize_labels(voted)

    def ensemble_full(
        self,
        preds: Sequence[Mapping],
        weights: Optional[Sequence[float]] = None,
    ) -> Dict:
        rubric = self.ensemble_rubric(preds, weights)
        errors = self.ensemble_errors(preds)
        tot = total_score(rubric)
        return {
            "rubric": rubric,
            "total_score": tot,
            "taxonomy_error": errors,
        }