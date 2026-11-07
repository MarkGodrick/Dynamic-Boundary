# Configuring an experiment

Run commands from the repository root. Search uses `scripts/run_sgbs.py`; adaptive evaluation uses `main.py` and an anchor bundle. Both entry points accept a YAML configuration and `--models` to override its model list.

## Python environments

| Use | Dependencies and setup |
|---|---|
| API inference, search, and adaptive evaluation | Python 3.11 and `requirements.txt` |
| Tests | `requirements-dev.txt` |
| Local Hugging Face inference | `requirements-hf.txt` plus a PyTorch build for the machine |
| Plotting and embedding-based analyses | `requirements-analysis.txt` |
| Local vLLM inference | A separate conda environment named by `target_model.conda_env` or the model's panel entry |

The vLLM backend starts an inference server as a subprocess. Model presets in `core/panel_config.py` specify environment names, precision, and launch arguments. The research setup used environments named `vllm085` and `vllm`; create the matching environment or change `conda_env` to your installation. `requirements*.txt` describes the main process and analysis dependencies.

The instruction-following verifiers use NLTK resources. They download missing resources on first import. To prepare them ahead of time:

```bash
python -m nltk.downloader \
  -d skills/constrained_completion/verifiers/ifbench/.nltk_data \
  punkt punkt_tab stopwords averaged_perceptron_tagger_eng
```

## API targets

The examples use an OpenAI-compatible endpoint at `http://localhost:8000/v1`. Set `base_url` to your endpoint. `api_key_env` names the environment variable from which the client reads the key:

```yaml
target_model:
  type: api
  base_url: http://localhost:8000/v1
  api_key_env: TARGET_API_KEY
  max_new_tokens: 2048
  temperature: 1.0
```

```bash
export TARGET_API_KEY='your-api-key'
python scripts/run_sgbs.py --config configs/examples/search_api.yaml \
  --models your-model-id
```

Constrained instruction following uses `composer.type: template` and `judge.type: constraint`. Other categories' supplied configurations use LLM composition and judging; their `api_key_env` fields identify the credentials to export, commonly `DEEPSEEK_API_TOKEN`. The composer, judge, and target can use different endpoints and credentials.

## Local models and the reference panel

| Field | Meaning |
|---|---|
| `target_model.type` | Target backend: `api`, `hf`, or `vllm` |
| `target_model.name` / `models` | Model identifier; `--models` overrides the YAML model list |
| `cuda_devices` | Comma-separated visible GPU IDs |
| `conda_env` | Conda environment containing the vLLM executable |
| `extra_args` | Additional vLLM server arguments, including parallelism and chat template |
| `gpu_memory_utilization` | Fraction of GPU memory available to the server |
| `max_new_tokens`, `max_model_len` | Generation cap and server context window |

For a model listed in `core/panel_config.py`, the dispatcher takes `extra_args`, `conda_env`, `dtype`, and `gpu_memory_utilization` from that entry. YAML controls GPU placement through `cuda_devices`. For other model identifiers, provide the backend settings in YAML.

The number of visible GPUs must equal tensor parallelism × data parallelism × pipeline parallelism. For example, a preset with `--tensor-parallel-size 2` and no other parallelism needs two visible GPU IDs. Configure GPU placement for the selected model before running a multi-model experiment.

Qwen3-8B's preset supplies `configs/model_chat_templates/qwen3_no_think_chat_template.jinja` to use the experiment's non-thinking prompt format. Keep this template with the preset when reproducing that condition. Larger models and longer contexts need additional memory or lower concurrency; `search.batch_size`, response-collection workers, and `step2.panel_workers` control different parts of the workload.

Category and model-specific generation budgets are defined in `core/inference_budgets.py`; the entry points and collection scripts apply these where appropriate. For sycophancy resistance, generation limits apply per turn.

## Adaptive evaluation

| Setting | Meaning |
|---|---|
| `adaptive.estimator` | `eap` or `laplace` posterior estimator |
| `adaptive.prior_mu`, `prior_sigma`, `n_quad` | Prior and EAP integration settings |
| `adaptive.N` | Target trials for an anchor item |
| `adaptive.pilot_quantiles` | Initial anchor locations across the difficulty bank |
| `adaptive.stab_eps`, `stab_K` | Posterior-stability tolerance and history length |
| `adaptive.tau_pad`, `t_min`, `hysteresis_L` | Coverage and Step-2 trigger settings |
| `adaptive.max_iters` | Maximum adaptive iterations |
| `step2.K` | New boundary items requested during augmentation |
| `step2.sgbs_config` | Search configuration used by Step 2 |
| `step2.panel_workers` | Reference-panel scoring concurrency |

Even with an API target, Step 2 runs the local reference panel to calibrate new items. Configure that panel's model environments and GPU placement when using the full adaptive workflow.

```bash
python main.py --config configs/examples/adaptive_api.yaml \
  --anchor-bundle benchmarks/constrained_completion/anchor_alo_bundle_M9.json \
  --models your-model-id
```

## Outputs and resuming

`experiment.output_dir` contains per-model run directories. Search saves accepted boundary items, evaluations, and `search_state.json`; adaptive evaluation saves `result.json` and its evaluation trace. Repeating a command skips a completed run. Search resumes a matching partial run; adaptive evaluation starts an interrupted run again. `--force-fresh` starts a new run directory. Consult each script's `--help` for its additional options.
