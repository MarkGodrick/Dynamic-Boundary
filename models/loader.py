"""Factory for creating target model instances from config."""

from logger import get_logger
from .response_parser import get_parser

logger = get_logger(__name__)


def create_target_model(config: dict):
    """Create a TargetModel from config dict.

    Config keys:
        type: "hf", "api", "external_api", or "vllm" (default: "hf")
        name: model name or HuggingFace ID
        device: e.g. "cuda:0" (for hf)
        dtype: "bf16", "fp16", "fp32" (for hf)
        max_new_tokens: generation length limit
        temperature: sampling temperature
        api_key_env, base_url: (for api)
        api_key_env, providers, provider, max_concurrency: (for external_api)
        conda_env, cuda_devices, port, gpu_memory_utilization,
        max_model_len, extra_args, server_timeout: (for vllm)
    """
    model_type = config.get("type", "hf")

    if model_type == "hf":
        from .hf_model import HFModel

        parser = get_parser(config.get("name", ""))
        return HFModel(
            model_name=config["name"],
            device=config.get("device", "cuda:0"),
            dtype=config.get("dtype", "bf16"),
            max_new_tokens=config.get("max_new_tokens", 512),
            temperature=config.get("temperature", 1.0),
            response_parser=parser,
        )

    if model_type == "api":
        from .api_model import APIModel

        parser = get_parser(config.get("name", ""))
        return APIModel(
            model=config["name"],
            api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
            base_url=config.get("base_url"),
            max_retries=config.get("max_retries", 5),
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_new_tokens", 512),
            response_parser=parser,
        )

    if model_type == "external_api":
        from .external_api_model import ExternalAPIModel

        parser = get_parser(config.get("name", ""))
        return ExternalAPIModel(
            model=config["name"],
            api_key_env=config.get("api_key_env", "AICODEMIRROR_API_KEY"),
            providers=config.get("providers", {}),
            provider=config.get("provider"),
            max_retries=config.get("max_retries", 5),
            retry_delay=config.get("retry_delay", 1.0),
            temperature=config.get("temperature"),
            max_tokens=config.get("max_new_tokens", 2048),
            max_concurrency=config.get("max_concurrency", 4),
            timeout=config.get("timeout", 600.0),
            response_parser=parser,
        )

    if model_type == "vllm":
        from .vllm_model import VLLMModel

        return VLLMModel(
            model_name=config["name"],
            conda_env=config.get("conda_env", "vllm085"),
            cuda_devices=config.get("cuda_devices", "0"),
            port=config.get("port", 8100),
            dtype=config.get("dtype", "bfloat16"),
            max_model_len=config.get("max_model_len", 8192),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_new_tokens", 1024),
            extra_args=config.get("extra_args", []),
            server_timeout=config.get("server_timeout", 600),
        )

    raise ValueError(f"Unknown target_model type: {model_type}")


def unload_model(model) -> None:
    """Free GPU memory from a model (HF or vLLM)."""
    if hasattr(model, "shutdown"):
        model.shutdown()
        logger.info("vLLM server shut down")
        return

    import torch

    if hasattr(model, "model"):
        del model.model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded, GPU memory freed")
