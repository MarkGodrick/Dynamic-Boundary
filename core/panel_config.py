"""Reference-panel and additional target-model launch settings.

Entries specify model identifiers, GPU placement, conda environments, and
vLLM arguments. The dispatcher merges these with experiment YAML settings;
see docs/CONFIGURATION.md for precedence and parallelism requirements.
Calibrated panel abilities are stored in each category's anchor bundle.
Qwen3-8B uses the bundled non-thinking chat template; the Qwen2.5 AWQ
preset uses float16 for the AWQ-Marlin backend.
"""

# Nine reference models used for anchor calibration.
PANEL_MODELS = [
    {
        "name": "Qwen/Qwen2.5-72B-Instruct-AWQ",
        "cuda_devices": "0,1",
        "conda_env": "vllm085",
        "dtype": "float16",
        "gpu_mem_util": 0.9,
        "extra_args": [
            "--tensor-parallel-size", "2",
            "--quantization", "awq_marlin",
        ],

    },
    {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "cuda_devices": "0,1",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager", "--tensor-parallel-size", "2"],

    },
    {
        "name": "google/gemma-3-27b-it",
        "cuda_devices": "0,1",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager", "--tensor-parallel-size", "2"],

    },
    {
        "name": "Qwen/Qwen3-8B",
        "cuda_devices": "0",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": [
            "--enforce-eager",
            "--chat-template",
            "configs/model_chat_templates/qwen3_no_think_chat_template.jinja",
        ],
    },
    {
        "name": "google/gemma-3-12b-it",
        "cuda_devices": "0,1",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": ["--tensor-parallel-size", "2"],

    },
    {
        "name": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "cuda_devices": "0,1",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager", "--tensor-parallel-size", "2"],

    },
    {
        "name": "google/gemma-3-4b-it",
        "cuda_devices": "0,1",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": ["--tensor-parallel-size", "2"],

    },
    {
        "name": "mistralai/Mistral-Nemo-Instruct-2407",
        "cuda_devices": "0,1",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": ["--tensor-parallel-size", "2"],

    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "cuda_devices": "0,1",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": ["--tensor-parallel-size", "2"],

    },
]

# Additional target models used for held-out evaluation.
VALIDATION_MODELS = [
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "cuda_devices": "0,1,2,3",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager", "--tensor-parallel-size", "4"],
    },
    {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "cuda_devices": "0,1,2,3",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager", "--tensor-parallel-size", "4"],
    },
    {
        "name": "openai/gpt-oss-20b",
        "cuda_devices": "0",
        "conda_env": "vllm",
        "gpu_mem_util": 0.85,
        "extra_args": ["--enforce-eager"],
    },

    {
        "name": "microsoft/phi-4",
        "cuda_devices": "0,1,2,3",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": [
            "--enforce-eager",
            "--tensor-parallel-size", "2",

        ],
    },
    {
        "name": "allenai/Olmo-3-7B-Instruct",
        "cuda_devices": "0,1,2,3",
        "conda_env": "vllm",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager", "--data-parallel-size", "2"],
    },

    {
        "name": "microsoft/Phi-4-reasoning-plus",
        "cuda_devices": "0,1,2,3",
        "conda_env": "vllm085",
        "gpu_mem_util": 0.9,
        "extra_args": [
            "--enforce-eager",
            "--tensor-parallel-size", "2",

        ],
    },
    {
        "name": "ibm-granite/granite-4.1-8b",
        "cuda_devices": "0,1,2,3",
        "conda_env": "vllm",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager", "--data-parallel-size", "2"],
    },
    {
        "name": "tiiuae/Falcon-H1R-7B",
        "cuda_devices": "0,1,2,3",
        "conda_env": "vllm",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager", "--data-parallel-size", "4"],
    },

    {
        "name": "allenai/Olmo-3.1-32B-Instruct",
        "cuda_devices": "0,1,2,3",
        "conda_env": "vllm",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager", "--data-parallel-size", "2"],
    },
    {
        "name": "openai/gpt-oss-120b",
        "cuda_devices": "0",
        "conda_env": "vllm",
        "gpu_mem_util": 0.9,
        "extra_args": ["--enforce-eager"],
    },
]

# Optional models for panel-selection profiling.
CANDIDATE_MODELS = [
]
