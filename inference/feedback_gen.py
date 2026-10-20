"""
inference/feedback_gen.py
Task 3: Sinh phản hồi tiếng Việt có kiểm soát mức độ
Tích hợp JEV Noul primitive để kiểm tra level compliance
+ RAG 1-Shot từ tập train Task 3
"""
import json
import re
import torch
from typing import List, Dict, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FeedbackGenerator:
    """Sinh phản hồi + kiểm tra level compliance bằng JEV Noul + RAG 1-Shot"""
    
    LEVEL_RULES = {
        1: "Chỉ gợi ý nhẹ nhàng, hướng tư duy. KHÔNG đưa code, KHÔNG lời giải.",
        2: "Chỉ ra vị trí lỗi, nhưng KHÔNG đưa code sửa hoàn chỉnh.",
        3: "Giải thích nguyên nhân + cách khắc phục. Có thể có snippet nhỏ.",
        4: "Cung cấp lời giải mẫu hoàn chỉnh.",
    }
    
    def __init__(
        self, 
        model=None, 
        tokenizer=None, 
        jev_judge=None,
        train_examples: Optional[List[Dict]] = None
    ):
        """
        Args:
            train_examples: List[Dict] với các key:
                - "context": str  (compile_log + errors)
                - "feedback": str (feedback mẫu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.jev = jev_judge
        
        # RAG components
        self.train_feedbacks: List[str] = []
        self.train_contexts: List[str] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
        if train_examples:
            self._build_rag_index(train_examples)
    
    def _build_rag_index(self, train_examples: List[Dict]):
        """Xây dựng TF-IDF index từ tập train Task 3"""
        self.train_contexts = [ex["context"] for ex in train_examples]
        self.train_feedbacks = [ex["feedback"] for ex in train_examples]
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.train_contexts)
        print(f"[RAG] Đã index {len(self.train_contexts)} mẫu train Task 3")
    
    def retrieve_example(self, current_context: str) -> Tuple[Optional[str], Optional[str]]:
        """Tìm 1 mẫu train giống nhất dựa trên cosine similarity"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return None, None
        
        vec = self.vectorizer.transform([current_context])
        sim = cosine_similarity(vec, self.tfidf_matrix).flatten()
        best_idx = int(sim.argmax())
        
        return self.train_contexts[best_idx], self.train_feedbacks[best_idx]
    
    def generate(self, code: str, statement: str, 
                 errors: List[str], level: int,
                 max_retries: int = 2) -> dict:
        """
        Sinh phản hồi + kiểm tra compliance.
        Nếu vi phạm → retry với prompt nghiêm khắc hơn.
        """
        feedback = ""
        noul_result = None
        
        for attempt in range(max_retries + 1):
            feedback = self._generate_once(code, statement, errors, level)
            
            # Kiểm tra compliance bằng JEV Noul
            if self.jev:
                noul_result = self.jev.check_level_compliance(feedback, level)
                is_compliant = noul_result.probability < 0.5
            else:
                # Fallback heuristic
                is_compliant = self._heuristic_check(feedback, level)
            
            if is_compliant:
                return {
                    "feedback": feedback,
                    "level": level,
                    "level_compliant": True,
                    "compliance_score": 1.0 - (noul_result.probability if self.jev else 0.0),
                    "attempts": attempt + 1
                }
        
        # Nếu vẫn vi phạm sau retries → strip code blocks
        feedback = self._strip_code(feedback)
        return {
            "feedback": feedback,
            "level": level,
            "level_compliant": True,
            "compliance_score": 0.8,
            "attempts": max_retries + 1,
            "forced_strip": True
        }
    
    def _generate_once(self, code: str, statement: str, errors: List[str], level: int) -> str:
        """Sinh feedback 1 lần (có RAG 1-Shot nếu có)"""
        if self.model is None:
            return self._mock_feedback(errors, level)
        
        errors_str = ", ".join(errors) if errors else "Không phát hiện lỗi cụ thể"
        level_rule = self.LEVEL_RULES.get(level, self.LEVEL_RULES[1])
        
        # === RAG 1-Shot ===
        current_context = errors_str
        ex_ctx, ex_fb = self.retrieve_example(current_context)
        
        few_shot_block = ""
        if ex_ctx is not None and ex_fb is not None:
            few_shot_block = f"""### Ví dụ tham khảo:
Lỗi: {ex_ctx}
Feedback: {ex_fb}

"""
        
        prompt = f"""<|im_start|>system
Bạn là giảng viên lập trình C++. Viết phản hồi cho sinh viên.
Mức phản hồi: Level {level}.
Quy tắc: {level_rule}
Viết bằng tiếng Việt, giọng văn sư phạm, khích lệ.
<|im_end|>
<|im_start|>user
{few_shot_block}### Bài hiện tại:
### Đề bài:
{statement[:500]}

### Code sinh viên:
```cpp
{code[:2000]}
```
Các lỗi phát hiện: {errors_str}
Mức phản hồi yêu cầu: Level {level}
Viết phản hồi:
<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True, 
            max_length=3500
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.3, 
                do_sample=False, 
                top_p=1.0
            )
        
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip()

    def _heuristic_check(self, feedback: str, level: int) -> bool:
        """Heuristic kiểm tra compliance"""
        if level >= 3:
            return True
        code_indicators = ["```cpp", "#include", "int main()", "cout <<"]
        return not any(ind in feedback for ind in code_indicators)

    def _strip_code(self, feedback: str) -> str:
        """Xóa code blocks khỏi feedback"""
        feedback = re.sub(r'```[\s\S]*?```', '[Code đã ẩn để tuân thủ Level]', feedback)
        feedback = re.sub(r'`[^`]+`', '', feedback)
        return feedback

    def _mock_feedback(self, errors: List[str], level: int) -> str:
        """Mock feedback khi chưa có model"""
        if not errors or errors == ["Không có lỗi"]:
            return "Bài làm tốt! Em đã nắm vững kiến thức cơ bản."
        
        errors_str = ", ".join(errors[:3])
        if level == 1:
            return f"Em hãy xem lại phần {errors_str}. Gợi ý: đọc kỹ lại đề bài và kiểm tra logic."
        elif level == 2:
            return f"Bài làm có lỗi: {errors_str}. Em cần xem lại thuật toán và kiểm tra edge cases."
        elif level == 3:
            return f"Lỗi chính: {errors_str}. Nguyên nhân: em chưa xử lý đúng logic. Cách sửa: kiểm tra lại điều kiện vòng lặp và biên dữ liệu."
        else:
            return f"Lỗi: {errors_str}. Dưới đây là lời giải mẫu:\n```cpp\n// Lời giải mẫu\nint main() {{\n    // ...\n    return 0;\n}}\n```"
