"""Response parsers for different model output formats.

Adapted from Numerical-Alignment/benchmarks/utils.py:clean_reasoning_response().
"""


class ResponseParser:
    """Base parser — returns raw text unchanged."""

    def parse(self, text: str) -> str:
        return text

    def parse_for_judge(self, text: str) -> str:
        """Parse response for judge consumption.

        By default same as parse(). Subclasses can override to include
        extra context (e.g., thinking traces for harmful content detection).
        """
        return self.parse(text)

    def parse_with_reasoning(self, text: str) -> tuple[str, str | None]:
        """Return (final_response, reasoning_content_or_none).

        The reasoning slot is for the **stripped CoT** (what `parse()` removed).
        Base class does no stripping → reasoning is always None.

        Subclasses that strip (R1ResponseParser, HarmonyResponseParser) return
        the stripped CoT text so callers can persist it alongside the final
        response instead of losing it to the parser.
        """
        return (self.parse(text), None)


class R1ResponseParser(ResponseParser):
    """Strip <think>...</think> reasoning traces from DeepSeek-R1 output.

    For harmful_refusal judging, the thinking trace IS relevant (model might
    reason about harmful content in <think> even if final answer refuses).
    So parse_for_judge() keeps the full text while parse() strips thinking.
    """

    def parse(self, text: str) -> str:
        if "</think>" in text:
            post_think = text.rsplit("</think>", 1)[1].lstrip()
            if post_think:
                return post_think
        return text

    def parse_for_judge(self, text: str) -> str:
        # For harmful content judging, include thinking trace
        return text

    def parse_with_reasoning(self, text: str) -> tuple[str, str | None]:
        """Split on </think>: content before is reasoning, after is final.

        Returns (text, None) if:
          - </think> absent (truncated mid-CoT, or no reasoning emitted), OR
          - post-think is empty (model stopped right at </think>)
        In both cases parse() returns raw text too, keeping semantics aligned.
        """
        if "</think>" not in text:
            return (text, None)
        pre, post = text.rsplit("</think>", 1)
        post = post.lstrip()
        if not post:
            return (text, None)
        reasoning = pre.lstrip()
        # Strip optional leading <think> opener (vLLM chat template may or
        # may not expose it as output; handle both cases).
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>"):]
        reasoning = reasoning.strip()
        return (post, reasoning or None)


class HarmonyResponseParser(ResponseParser):
    """Parse gpt-oss harmony response format.

    Handles two variants:
    1. Raw harmony with special tokens preserved:
       <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...
    2. Harmony traces rendered into plain text (special tokens stripped):
       analysis ... assistantfinal ...
    """

    def parse(self, text: str) -> str:
        cleaned = text

        # Raw harmony format: extract content after <|channel|>final<|message|>
        if "<|channel|>final" in cleaned and "<|message|>" in cleaned:
            post_final = cleaned.rsplit("<|channel|>final", 1)[1]
            if "<|message|>" in post_final:
                post_final = post_final.split("<|message|>", 1)[1]
                stop_markers = ["<|return|>", "<|end|>", "<|call|>"]
                stop_positions = [
                    post_final.find(m) for m in stop_markers if m in post_final
                ]
                if stop_positions:
                    post_final = post_final[: min(stop_positions)]
                post_final = post_final.lstrip()
                if post_final:
                    return post_final

        # Plain text harmony: extract content after "assistantfinal"
        if "assistantfinal" in cleaned:
            post_final = cleaned.rsplit("assistantfinal", 1)[1].lstrip()
            if post_final:
                return post_final

        return cleaned

    def parse_for_judge(self, text: str) -> str:
        # For judging, include analysis channel too (may contain harmful reasoning)
        return text

    def parse_with_reasoning(self, text: str) -> tuple[str, str | None]:
        """Extract final channel as response, analysis channel as reasoning.

        Handles the two Harmony variants:
          1. Raw (with special tokens): <|channel|>analysis<|message|>...<|end|>
          2. Plain-text (tokens stripped): analysis ... assistantfinal ...

        If neither variant matches, reasoning is None and parse() fallthrough
        returns the text unchanged.

        Note: this is a simplified extraction — tool-call (`commentary`) and
        multi-analysis-block edge cases are ignored. gpt-oss is not yet in
        our panel; upgrade this when we enable it.
        """
        final = self.parse(text)
        reasoning: str | None = None

        # Raw harmony
        if "<|channel|>analysis" in text and "<|message|>" in text:
            idx = text.find("<|channel|>analysis")
            after = text[idx:]
            if "<|message|>" in after:
                after = after.split("<|message|>", 1)[1]
                stop_markers = ["<|end|>", "<|channel|>", "<|start|>", "<|return|>"]
                stop_positions = [after.find(m) for m in stop_markers if m in after]
                if stop_positions:
                    after = after[: min(stop_positions)]
                reasoning = after.strip() or None

        # Plain-text harmony
        if reasoning is None and "assistantfinal" in text:
            pre, _ = text.split("assistantfinal", 1)
            pre = pre.strip()
            if pre.lower().startswith("analysis"):
                pre = pre[len("analysis"):].strip()
            reasoning = pre or None

        return (final, reasoning)


def get_parser(model_name: str) -> ResponseParser:
    """Auto-detect the appropriate parser based on model name."""
    name_lower = model_name.lower()
    if "deepseek-r1" in name_lower or "r1-distill" in name_lower:
        return R1ResponseParser()
    if "gpt-oss" in name_lower or "harmony" in name_lower:
        return HarmonyResponseParser()
    return ResponseParser()
