# Dynamic-Boundary: Dynamic Boundary Evaluation

Code for **Beyond Fixed Benchmarks and Worst-Case Attacks: Dynamic Boundary Evaluation for Language Models**, by Haoxiang Wang, Da Yu, and Huishuai Zhang. NeurIPS 2026 Evaluations & Datasets Track.

[Paper](https://arxiv.org/abs/2605.06213) · [DBE-Bench](https://huggingface.co/datasets/AItherapist/DBE-Bench) · [Code guide](docs/CODE_GUIDE.md) · [Configuration](docs/CONFIGURATION.md) · [Reproducing experiments](scripts/REPRODUCIBILITY.md)

Dynamic Boundary Evaluation (DBE) measures where a language model transitions between success and failure. This repository implements Skill-Guided Boundary Search (SGBS), Binomial Rasch calibration, and adaptive evaluation on the calibrated difficulty scale.

| Category | Directory name | Evaluated behavior | ALO anchors |
|---|---|---|---:|
| Harmful request refusal | `harmful_refusal` | Refusing harmful requests under jailbreak compositions | 293 |
| Over-refusal | `over_refusal` | Answering benign requests that resemble harmful ones | 299 |
| Constrained instruction following | `constrained_completion` | Satisfying combinations of verifiable constraints | 183 |
| Sycophancy resistance | `sycophancy_resistance` | Maintaining a correct answer under persuasive, multi-turn pressure | 416 |

Each category has its own calibrated difficulty scale. The reference panel contains nine models; ALO selects an item when at least one panel model's empirical success rate lies in `[0.20, 0.80]`.

## Installation and data

Use Python 3.11 for the main process:

```bash
python -m pip install -r requirements.txt
```

Accept the access conditions on [DBE-Bench](https://huggingface.co/datasets/AItherapist/DBE-Bench), then authenticate and download the ALO version:

```bash
hf auth login
python scripts/download_data.py --revision v1.1.0-alo --with-responses
```

The downloader checks file hashes and places seed requests in `data/`, skill dictionaries in `skills/`, calibrated bundles and evaluation sets in `benchmarks/`, and aggregate response counts in `results/`. Omit `--with-responses` when only running search or adaptive evaluation. For an existing local dataset copy, use `--source-dir /path/to/DBE-Bench` instead of `--revision`.

API targets use OpenAI-compatible endpoints. Local Hugging Face inference also uses PyTorch and `requirements-hf.txt`; vLLM inference uses a separate conda environment. See [Configuration](docs/CONFIGURATION.md) for backend setup, credentials, and GPU settings.

## Run an experiment

The API examples use constrained instruction following, with a template composer and rule-based judge. Set `target_model.base_url` in the example YAML to your endpoint and replace `your-model-id` with its served model name.

```bash
export TARGET_API_KEY='your-api-key'

# Find prompts near the target model's success/failure boundary.
python scripts/run_sgbs.py --config configs/examples/search_api.yaml \
  --models your-model-id

# Estimate ability on the calibrated anchor bank and augment when coverage fails.
python main.py --config configs/examples/adaptive_api.yaml \
  --anchor-bundle benchmarks/constrained_completion/anchor_alo_bundle_M9.json \
  --models your-model-id
```

Adaptive evaluation starts with anchor-based computerized adaptive testing (CAT). When coverage fails, Step 2 runs SGBS and evaluates the new items on the reference panel to place them on the same scale. This step uses the local panel settings in `core/panel_config.py`, including its vLLM environments and GPUs. `step2.K` sets the number of new boundary items; the example uses 50.

Category-specific search and adaptive configurations are under `configs/<category>/`. Results are written to `experiment.output_dir`; completed runs are skipped. Search can resume partial runs; interrupted adaptive runs restart. Use `--force-fresh` to start another run.

## Find your way around

| Path | Purpose |
|---|---|
| `main.py`, `adaptive/` | CAT, posterior updates, coverage checks, and Step-2 calibration |
| `scripts/run_sgbs.py`, `search/` | Skill selection, prompt composition, and boundary search |
| `difficulty/` | Rasch estimation and alternative IRT models |
| `models/`, `judges/` | Target-model backends and success/failure scoring |
| `core/` | Data loading, model dispatch, reference-panel settings, and generation budgets |
| `configs/` | Experiment configurations and API examples |
| `scripts/analysis/`, `scripts/data_prep/` | Experiment analysis and dataset construction |
| `tests/` | Tests with synthetic fixtures and mocked model backends |

The [code guide](docs/CODE_GUIDE.md) follows an item through the pipeline. The [reproducibility guide](scripts/REPRODUCIBILITY.md) maps the paper's experiments to their scripts and inputs, with commands for analyses using the published response counts.

To run the tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## License and citation

Original code is licensed under MIT. Bundled IFEval and IFBench verifiers use Apache-2.0; the DeepSeek tokenizer retains its upstream license files. See [Third-party notices](THIRD_PARTY_NOTICES.md). DBE-Bench has its own dataset license and access terms.

```bibtex
@misc{wang2026fixedbenchmarksworstcaseattacks,
      title={Beyond Fixed Benchmarks and Worst-Case Attacks: Dynamic Boundary Evaluation for Language Models},
      author={Haoxiang Wang and Da Yu and Huishuai Zhang},
      year={2026},
      eprint={2605.06213},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.06213},
}
```
