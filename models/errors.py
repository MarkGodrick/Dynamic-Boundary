"""Custom exceptions for target model errors."""


class ContextOverflowError(Exception):
    """Raised when a multi-turn conversation cannot fit within the model's
    context window. Used by the evaluator to gracefully end the conversation
    early instead of crashing.
    """
