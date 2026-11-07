"""Generation and context budgets by category and model.

max_tokens caps each target-model response; max_model_len sets the vLLM
context window for the prompt, response, and multi-turn history. Sycophancy
resistance uses a per-turn generation cap. Model-specific overrides provide
larger budgets for reasoning models. Lookup helpers prefer model overrides,
then category settings where available, then the caller's default.
"""

from typing import Optional


CATEGORY_MAX_TOKENS = {
    "harmful_refusal": 2048,
    "over_refusal": 3072,
    "constrained_completion": 2048,
    "sycophancy_resistance": 2048,
}

# Per-model max_tokens overrides keyed by (category, model_id). model_id is
# the HF id used in configs (e.g., "tiiuae/Falcon-H1R-7B"), NOT the sanitized
# directory name. SR values apply per-turn.
MODEL_MAX_TOKENS = {
    ("harmful_refusal",         "tiiuae/Falcon-H1R-7B"):       4096,
    ("over_refusal",            "tiiuae/Falcon-H1R-7B"):       8192,
    ("constrained_completion",  "tiiuae/Falcon-H1R-7B"):       8192,
    ("sycophancy_resistance",   "tiiuae/Falcon-H1R-7B"):      16384,

    ("harmful_refusal",         "microsoft/Phi-4-reasoning-plus"):  8192,
    ("over_refusal",            "microsoft/Phi-4-reasoning-plus"):  8192,
    ("constrained_completion",  "microsoft/Phi-4-reasoning-plus"):  8192,
    ("sycophancy_resistance",   "microsoft/Phi-4-reasoning-plus"):  8192,
}

# Default vLLM KV-cache window. Matches yaml configs and VLLMModel default.
DEFAULT_MAX_MODEL_LEN = 8192

# Per-model max_model_len overrides keyed by (category, model_id). Pick the
# smallest value that comfortably holds prompt + (multi-turn) history +
# max_tokens. SR rows hold prompt + 3 × per-turn max_tokens.
MODEL_MAX_MODEL_LEN = {
    ("harmful_refusal",         "microsoft/Phi-4-reasoning-plus"):  16384,
    ("over_refusal",            "microsoft/Phi-4-reasoning-plus"):  16384,
    ("constrained_completion",  "microsoft/Phi-4-reasoning-plus"):  16384,
    ("sycophancy_resistance",   "microsoft/Phi-4-reasoning-plus"):  32768,  # native 32k cap

    ("over_refusal",            "tiiuae/Falcon-H1R-7B"):  16384,
    ("constrained_completion",  "tiiuae/Falcon-H1R-7B"):  16384,
    ("sycophancy_resistance",   "tiiuae/Falcon-H1R-7B"):  65536,
}


def get_max_tokens(
    category: str,
    model: Optional[str] = None,
    default: int = 1024,
) -> int:
    """Return max_new_tokens for (category, model), preferring model-specific.

    Lookup order:
      1. MODEL_MAX_TOKENS[(category, model)]   if model given and matched
      2. CATEGORY_MAX_TOKENS[category]
      3. default
    """
    if model is not None:
        v = MODEL_MAX_TOKENS.get((category, model))
        if v is not None:
            return v
    return CATEGORY_MAX_TOKENS.get(category, default)


def get_max_model_len(
    category: str,
    model: Optional[str] = None,
    default: int = DEFAULT_MAX_MODEL_LEN,
) -> int:
    """Return vLLM max_model_len for (category, model), preferring model-specific.

    Lookup order:
      1. MODEL_MAX_MODEL_LEN[(category, model)]   if model given and matched
      2. default                                  (DEFAULT_MAX_MODEL_LEN = 8192)
    """
    if model is not None:
        v = MODEL_MAX_MODEL_LEN.get((category, model))
        if v is not None:
            return v
    return default
