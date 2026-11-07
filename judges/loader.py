"""Factory for creating judge instances from config."""

from logger import get_logger

logger = get_logger(__name__)


def create_judge(config: dict, category: str = "harmful_refusal"):
    """Create a Judge from config dict.

    Config keys:
        type: "keyword" or "llm_api" (default: "keyword")
        model, api_key_env, base_url: (for llm_api)
        max_retries, temperature: (for llm_api)
    """
    judge_type = config.get("type", "keyword")

    if judge_type == "keyword":
        from .keyword_judge import KeywordJudge
        logger.info("Using KeywordJudge")
        return KeywordJudge()

    if judge_type == "llm_api":
        from .llm_judge import LLMJudge
        judge = LLMJudge(
            model=config.get("model", "deepseek-chat"),
            api_key_env=config.get("api_key_env", "DEEPSEEK_API_TOKEN"),
            base_url=config.get("base_url", "https://api.deepseek.com"),
            category=category,
            max_retries=config.get("max_retries", 5),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 1024),
            extra_body=config.get("extra_body"),
            request_kwargs=config.get("request_kwargs"),
        )
        logger.info("Using LLMJudge (model=%s, category=%s)", config.get("model"), category)
        return judge

    if judge_type == "constraint":
        from .constraint_judge import ConstraintJudge
        logger.info("Using ConstraintJudge (rule-based)")
        return ConstraintJudge()

    raise ValueError(f"Unknown judge type: {judge_type}")
