"""Logit distribution over skill counts with directional feedback."""

import numpy as np


def init_skill_logits(
    max_count: int = 4, bare_request_bias: float = 3.0
) -> list[float]:
    """Initialize skill count logits with bias toward bare request (count=0)."""
    logits = [0.0] * max_count
    logits[0] = bare_request_bias
    return logits


def sample_count(
    logits: list[float],
    mis_size: int,
    top_k: int = 3,
    top_p: float = 0.9,
    lower_bound: int = 0,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample a skill count from the logit distribution.

    Uses dynamic window + softmax + top-k/top-p sampling.

    Args:
        lower_bound: Minimum skill count (e.g., 1 to exclude bare requests).
        rng: numpy Generator for deterministic sampling. Falls back to
             global np.random if None.
    """
    # Dynamic window: [lower_bound, min(argmax + 3, mis_size, len-1)]
    finite_logits = [l if l != float("-inf") else -1e9 for l in logits]
    peak = int(np.argmax(finite_logits[lower_bound:])) + lower_bound
    upper = min(peak + 3, mis_size, len(logits) - 1)

    window = logits[lower_bound: upper + 1]
    probs = _softmax(window)
    probs = _top_k_top_p_filter(probs, top_k, top_p)

    # Sample from filtered distribution, offset by lower_bound
    indices = np.arange(len(probs))
    if rng is not None:
        return int(rng.choice(indices, p=probs)) + lower_bound
    return int(np.random.choice(indices, p=probs)) + lower_bound


def update_logits(
    logits: list[float],
    count: int,
    result_type: str,
    alpha: float = 0.05,
    beta: float = 0.1,
    n_count: int = 1,
    lower_bound: int = 0,
) -> None:
    """Update logits in-place with directional feedback.

    Args:
        logits: The logit array to mutate.
        count: The skill count that was used.
        result_type: One of "edge_case", "partial", "too_easy", "too_hard", "no_combo".
        alpha: Failure decay coefficient.
        beta: Directional boost coefficient.
        n_count: Number of times this count has been sampled (for decay scaling).
        lower_bound: Minimum skill count — don't boost below this.
    """
    if result_type == "edge_case":
        logits[count] += 1.0
    elif result_type == "partial":
        logits[count] += 0.1
    elif result_type == "too_easy":
        logits[count] -= alpha / max(n_count, 1)
        if count + 1 < len(logits):
            logits[count + 1] += beta
    elif result_type == "too_hard":
        logits[count] -= alpha / max(n_count, 1)
        if count - 1 >= lower_bound:
            logits[count - 1] += beta
    elif result_type == "no_combo":
        logits[count] -= 1.0


def _softmax(values: list[float]) -> np.ndarray:
    """Numerically stable softmax over a list of logits."""
    arr = np.array(values, dtype=np.float64)
    arr = np.where(arr == float("-inf"), -1e9, arr)
    arr = arr - np.max(arr)
    exp_arr = np.exp(arr)
    return exp_arr / exp_arr.sum()


def _top_k_top_p_filter(
    probs: np.ndarray, top_k: int, top_p: float
) -> np.ndarray:
    """Apply top-k then top-p (nucleus) filtering, renormalize."""
    filtered = probs.copy()

    # Top-k: zero out everything except top-k entries
    if top_k < len(filtered):
        threshold = np.partition(filtered, -top_k)[-top_k]
        filtered[filtered < threshold] = 0.0

    # Top-p: sort descending, accumulate, zero out past threshold
    sorted_indices = np.argsort(-filtered)
    cumsum = 0.0
    for idx in sorted_indices:
        cumsum += filtered[idx]
        if cumsum > top_p:
            # Zero out remaining indices
            remaining = sorted_indices[list(sorted_indices).index(idx) + 1 :]
            filtered[remaining] = 0.0
            break

    # Renormalize
    total = filtered.sum()
    if total > 0:
        filtered /= total
    else:
        # Fallback: uniform over non-zero original entries
        nonzero = probs > 0
        filtered = np.where(nonzero, 1.0 / nonzero.sum(), 0.0)

    return filtered
