"""
inference/llm.py
Wrapper dùng chung cho Qwen2.5-Coder (+ LoRA adapter) — load một lần, dùng cho cả grader và feedback.

Điểm quan trọng:
- Không dùng `truncation=True` của tokenizer: cắt phải sẽ làm mất `<|im_start|>assistant` ở cuối prompt
  khiến model sinh rác. Thay vào đó `fit_prompt` rút gọn đề bài/code trước khi render.
- Greedy decoding, dừng ở `<|im_end|>` để tái lập được và nhanh hơn.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

MIN_STATEMENT_CHARS = 1200
_TRUNCATION_MARKER = "\n// ... [đã lược bớt do giới hạn độ dài] ...\n"


def shorten_head(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars)].rstrip() + "\n...[đã lược bớt]"


def shorten_middle(text: str, max_chars: int) -> str:
    """Giữ đầu (60%) và cuối (40%) của code — phần khai báo và phần cuối thường quan trọng nhất."""
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(_TRUNCATION_MARKER))
    head = int(keep * 0.6)
    return text[:head] + _TRUNCATION_MARKER + text[len(text) - (keep - head):]


class LLM:
    def __init__(self, base_model: str, adapter_path: Optional[str] = None,
                 max_input_tokens: int = 6000, load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if adapter_path and not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            raise FileNotFoundError(f"Không tìm thấy LoRA adapter hợp lệ tại '{adapter_path}'.")

        self.max_input_tokens = max_input_tokens
        print(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, device_map="auto",
            torch_dtype=torch.float16, trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel
            print(f"Loading LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        else:
            print("[WARN] Không có adapter -> chạy base model (zero-shot prompting baseline).")
        self.model = model.eval()
        self._torch = torch

        eos_ids = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
        self._eos_ids = sorted(i for i in eos_ids if isinstance(i, int) and i >= 0)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def fit_prompt(self, render: Callable[[str, str], str], statement: str, code: str) -> str:
        """Render prompt; nếu vượt budget thì rút gọn đề bài trước (giữ tối thiểu), sau đó tới code."""
        prompt = render(statement, code)
        n_tokens = self.count_tokens(prompt)
        for _ in range(8):
            if n_tokens <= self.max_input_tokens:
                break
            chars_per_token = len(prompt) / max(1, n_tokens)
            cut = int((n_tokens - self.max_input_tokens) * chars_per_token * 1.15) + 64
            statement_cut = min(cut, max(0, len(statement) - MIN_STATEMENT_CHARS))
            if statement_cut:
                statement = shorten_head(statement, len(statement) - statement_cut)
            if cut - statement_cut > 0:
                code = shorten_middle(code, max(200, len(code) - (cut - statement_cut)))
            prompt = render(statement, code)
            n_tokens = self.count_tokens(prompt)
        return prompt

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        from transformers import GenerationConfig

        config = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self._eos_ids,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with self._torch.no_grad():
            output = self.model.generate(**inputs, generation_config=config)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
