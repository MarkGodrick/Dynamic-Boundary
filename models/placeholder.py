"""Placeholder target model for testing the SGBS loop without GPU."""

import random


class PlaceholderTargetModel:
    """Returns random pass/fail responses based on a fixed pass_rate."""

    def __init__(self, pass_rate: float = 0.5):
        self.pass_rate = pass_rate

    def generate(self, prompt: str, seed: int | None = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.generate_conversation(messages, seed=seed)

    def generate_conversation(self, messages: list[dict], seed: int | None = None) -> str:
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        if rng.random() < self.pass_rate:
            return "I'm sorry, but I can't help with that request."
        last_content = messages[-1].get("content", "") if messages else ""
        return f"Sure, here is the information you asked for about: {last_content[:50]}"
