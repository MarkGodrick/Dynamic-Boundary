"""Judge factory for panel / filter inference pipelines.

Category-keyed factory: returns a `ConstraintJudge` for
`constrained_completion` and an `LLMJudge` (DeepSeek by default) for every
other category. Distinct from `judges.loader.create_judge`, which is
config-type-keyed and used by the interactive SGBS main loop.
"""


def create_judge(category: str, judge_config: dict):
    """Create the appropriate judge for the category."""
    if category == "constrained_completion":
        from judges.constraint_judge import ConstraintJudge
        return ConstraintJudge()
    else:
        from judges.llm_judge import LLMJudge
        return LLMJudge(
            model=judge_config.get("model", "deepseek-chat"),
            api_key_env=judge_config.get("api_key_env", "DEEPSEEK_API_TOKEN"),
            base_url=judge_config.get("base_url", "https://api.deepseek.com"),
            category=category,
            max_retries=judge_config.get("max_retries", 8),
            retry_delay=judge_config.get("retry_delay", 0.5),
            temperature=judge_config.get("temperature", 0.0),
            max_tokens=judge_config.get("max_tokens", 2048),
            extra_body=judge_config.get("extra_body"),
            request_kwargs=judge_config.get("request_kwargs"),
        )
