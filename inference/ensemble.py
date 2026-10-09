"""
inference/ensemble.py
Ensemble Qwen fine-tuned model + JEV judge
Chiến lược: Weighted ensemble dựa trên confidence
"""
import numpy as np
from typing import Dict, List, Optional
from collections import Counter


class GradingEnsemble:
    """
    Kết hợp kết quả từ nhiều judges.
    
    Strategy:
    - Rubric: weighted average theo confidence
    - Errors: union nếu >= threshold judges đồng ý
    - Compilable: conservative (min) — nếu 1 judge nói 0 → 0
    """
    
    RUBRIC_DIMS = ["compilable", "io_format", "logic", 
                   "edge_case", "complexity", "code_quality"]
    
    RUBRIC_RANGES = {
        "compilable": (0, 1), "io_format": (0, 1),
        "logic": (0, 4), "edge_case": (0, 2),
        "complexity": (0, 1), "code_quality": (0, 1)
    }
    
    def __init__(self, error_threshold: float = 0.5):
        self.error_threshold = error_threshold
    
    def ensemble_rubric(self, predictions: List[dict], 
                        weights: List[float] = None) -> Dict[str, int]:
        """
        Ensemble rubric scores.
        
        Args:
            predictions: list of {"rubric": {...}, ...}
            weights: trọng số cho mỗi prediction (default: đều)
        """
        if not predictions:
            return {d: 0 for d in self.RUBRIC_DIMS}
        
        n = len(predictions)
        if weights is None:
            weights = [1.0 / n] * n
        
        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
        result = {}
        for dim in self.RUBRIC_DIMS:
            lo, hi = self.RUBRIC_RANGES[dim]
            
            if dim == "compilable":
                # Conservative: nếu BẤT KỲ judge nào nói 0 → 0
                values = [p.get("rubric", {}).get(dim, 0) for p in predictions]
                result[dim] = min(values) if values else 0
            else:
                # Weighted average rồi round
                weighted_sum = sum(
                    w * p.get("rubric", {}).get(dim, 0) 
                    for w, p in zip(weights, predictions)
                )
                result[dim] = max(lo, min(hi, round(weighted_sum)))
        
        return result
    
    def ensemble_errors(self, predictions: List[dict], 
                        threshold: float = None) -> List[str]:
        """
        Ensemble error labels.
        Giữ nhãn nếu >= threshold fraction judges đồng ý.
        """
        if threshold is None:
            threshold = self.error_threshold
        
        if not predictions:
            return []
        
        n = len(predictions)
        label_counts = Counter()
        
        for pred in predictions:
            for label in pred.get("errors", []):
                label_counts[label] += 1
        
        min_count = max(1, int(threshold * n))
        result = [label for label, count in label_counts.items() 
                  if count >= min_count]
        
        # Nếu không có lỗi nào được chọn -> "Không có lỗi"
        if not result:
            result = ["Không có lỗi"]
        
        return sorted(result)
    
    def ensemble_full(self, qwen_pred: dict, jev_pred: dict,
                      qwen_weight: float = 0.6, 
                      jev_weight: float = 0.4) -> dict:
        """
        Ensemble Qwen + JEV.
        
        Qwen thường tốt hơn về logic analysis → weight cao hơn.
        JEV tốt hơn về structured consistency → weight thấp hơn nhưng quan trọng.
        """
        predictions = []
        weights = []
        
        if qwen_pred:
            predictions.append(qwen_pred)
            weights.append(qwen_weight)
        
        if jev_pred:
            predictions.append(jev_pred)
            weights.append(jev_weight)
        
        rubric = self.ensemble_rubric(predictions, weights)
        errors = self.ensemble_errors(predictions)
        total = sum(rubric.values())
        
        return {
            "rubric": rubric,
            "total_score": total,
            "errors": errors,
            "ensemble_info": {
                "sources": len(predictions),
                "weights": weights
            }
        }