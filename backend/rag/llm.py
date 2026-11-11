"""Local Qwen3-4B (MLX 4-bit) text generation via MLX-LM.

The model is loaded once per process and reused. Qwen3's "thinking" mode is
disabled: we want short, structured, deterministic outputs, and reasoning
traces would consume the limited context/time budget on a 16 GB Mac.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

from backend.config import get_settings

log = logging.getLogger(__name__)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    generation_tokens: int
    seconds: float
    generation_tps: float
    peak_memory_gb: float


class LocalLLM:
    def __init__(self, model_path: str):
        from mlx_lm import load

        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_path)
        self.model_path = model_path
        self._lock = threading.Lock()  # MLX generation is not re-entrant
        log.info("Loaded LLM from %s in %.1fs", model_path, time.perf_counter() - t0)

    def build_prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate(
        self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None
    ) -> GenerationResult:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        s = get_settings()
        max_tokens = max_tokens or s.llm_max_tokens
        temperature = s.llm_temperature if temperature is None else temperature
        prompt = self.build_prompt(messages)
        sampler = make_sampler(temp=temperature)

        with self._lock:
            t0 = time.perf_counter()
            pieces: list[str] = []
            last = None
            for resp in stream_generate(
                self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
            ):
                pieces.append(resp.text)
                last = resp
            elapsed = time.perf_counter() - t0

        text = THINK_BLOCK_RE.sub("", "".join(pieces)).strip()
        return GenerationResult(
            text=text,
            prompt_tokens=getattr(last, "prompt_tokens", 0),
            generation_tokens=getattr(last, "generation_tokens", 0),
            seconds=elapsed,
            generation_tps=getattr(last, "generation_tps", 0.0),
            peak_memory_gb=getattr(last, "peak_memory", 0.0),
        )


@lru_cache(maxsize=1)
def get_llm() -> LocalLLM:
    s = get_settings()
    if not (s.llm_path / "config.json").exists():
        raise FileNotFoundError(
            f"LLM not found at {s.llm_path}. Run `python scripts/download_models.py`."
        )
    return LocalLLM(str(s.llm_path))
