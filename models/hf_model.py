"""HuggingFace local model — supports both CausalLM and VLM (text-only).

Auto-detects model type via AutoConfig architectures:
  - Standard causal LM → AutoModelForCausalLM + AutoTokenizer
  - VLM (e.g., Gemma3)  → AutoModelForImageTextToText + AutoProcessor

Adapted from Numerical-Alignment/generation/model_loader.py and model_generator.py.
"""

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    LogitsProcessor,
)

from logger import get_logger
from .response_parser import ResponseParser, get_parser

logger = get_logger(__name__)

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_MULTIMODAL_KEYWORDS = ("ImageText", "ConditionalGeneration", "ForVision")


class _SanitizeLogitsProcessor(LogitsProcessor):
    """Clamp logits to prevent NaN/Inf from crashing multinomial sampling."""

    def __call__(self, _input_ids, scores):
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        return scores


def _is_multimodal(model_name: str) -> bool:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    architectures = getattr(config, "architectures", None) or []
    return any(
        any(kw in arch for kw in _MULTIMODAL_KEYWORDS) for arch in architectures
    )


class HFModel:
    """HuggingFace local model for text generation.

    Handles both standard causal LM and VLM (text-only mode) transparently.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "bf16",
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        response_parser: ResponseParser | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.response_parser = response_parser or get_parser(model_name)

        torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
        self.is_multimodal = _is_multimodal(model_name)

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "device_map": device,
        }

        if self.is_multimodal:
            logger.info("Detected multimodal model, using AutoProcessor + AutoModelForImageTextToText")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, **load_kwargs,
            )
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs,
            )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": self.temperature,
            "logits_processor": [_SanitizeLogitsProcessor()],
        }

        params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(
            "Loaded %s | multimodal=%s | dtype=%s | device=%s | %.1fM params",
            model_name, self.is_multimodal, dtype, device, params_m,
        )

    def _build_messages(self, prompt: str) -> list[dict]:
        """Build chat messages in the format appropriate for this model."""
        if self.is_multimodal:
            # VLM processors expect content as list of typed dicts
            return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return [{"role": "user", "content": prompt}]

    def _tokenize(self, messages: list[dict]) -> dict:
        """Apply chat template and tokenize."""
        if self.is_multimodal:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

    def generate(self, prompt: str, seed: int | None = None) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: User message text.
            seed: Random seed for reproducibility.

        Returns:
            Parsed model response string.
        """
        messages = self._build_messages(prompt)
        return self.generate_conversation(messages, seed=seed)

    def generate_conversation(self, messages: list[dict], seed: int | None = None) -> str:
        """Generate a response for a multi-turn conversation.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            seed: Random seed for reproducibility.

        Returns:
            Parsed model response string.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        inputs = self._tokenize(messages)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self._gen_kwargs)

        new_token_ids = output_ids[0, input_length:]
        raw_response = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        del inputs, output_ids, new_token_ids
        torch.cuda.empty_cache()

        return self.response_parser.parse(raw_response)

    def generate_raw(self, prompt: str, seed: int | None = None) -> str:
        """Generate and return the UNPARSED response (for judge with full context)."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        messages = self._build_messages(prompt)
        inputs = self._tokenize(messages)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self._gen_kwargs)

        new_token_ids = output_ids[0, input_length:]
        raw_response = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        del inputs, output_ids, new_token_ids
        torch.cuda.empty_cache()

        return raw_response
