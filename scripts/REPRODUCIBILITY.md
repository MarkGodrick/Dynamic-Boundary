# Reproducing experiments

This guide connects DBE's data, calibration, search, and evaluation experiments to their entry points. Run commands from the repository root. Replace `<category>` with `harmful_refusal`, `over_refusal`, `constrained_completion`, or `sycophancy_resistance`.

## Inputs and workflow

Download the ALO data and aggregate response counts:

```bash
python scripts/download_data.py --revision v1.1.0-alo --with-responses
```

| Input | What it contains | Examples of analyses |
|---|---|---|
| `benchmarks/<category>/anchor_alo_bundle_M9.json` | Prompts, calibrated difficulties, panel and filter metadata | Adaptive evaluation |
| Panel response matrices | Effective pass counts and trial counts on the 500-item random pool | Filter comparison, Rasch fitting, panel leave-one-out stability, JML/MML comparison |
| Validation response matrices | Aggregate counts on the ALO anchors and SGBS boundary sets | Count-based validation and boundary-location analyses |
| Per-trial inference artifacts collected by experiment scripts | Generated text, judging records, completion metadata, and usage | Response replay, truncation, token-budget and cost analyses |
| CAT and Step-2 run outputs | Target traces, new items, and panel calibration results | Adaptive trajectory and augmentation analyses |
| Human/judge annotation records | Response-level labels and judge decisions | Judge agreement and bias analyses |

The published dataset supplies the first three input types. Experiment scripts produce the response-level and adaptive-run artifacts used by the other analyses. Each analysis script's module docstring describes its inputs and outputs.

The restored panel-response directories are:

| Category | Panel matrix directory |
|---|---|
| Harmful refusal | `results/harmful_refusal/filter_ablation_raw_v2_judgefix/` |
| Over-refusal | `results/over_refusal/filter_ablation_raw_v2_judgefix/` |
| Constrained instruction following | `results/constrained_completion/filter_ablation_raw_v2/` |
| Sycophancy resistance | `results/sycophancy_resistance/filter_ablation_raw_v2/` |

Each model subdirectory contains `response_matrix.jsonl`. Rows identify an item with `prompt_id` and record `pass_count` and `total_trials`. ALO-anchor validation counts are under `results/<category>/anchor_set_alo/`; boundary-item panel counts are under `results/<category>/validation/edge_case_slot50/`.

## Analyses using the published counts

### Panel calibration stability

```bash
python scripts/analysis/claim1_calibration_stability.py \
  --anchor-filter alo --results-root results \
  --output results/analysis/claim1_calibration_stability_alo.json
```

For the parametric null comparison, add `--with-null-sim --n-bootstrap 100 --seed 42`. This analysis refits the bank with each reference model left out and measures difficulty displacement and rank stability.

### Anchor-filter comparison

```bash
python scripts/analysis/compare_filters.py \
  --response-dir results/constrained_completion/filter_ablation_raw_v2 \
  --output results/analysis/filter_comparison/constrained_completion.json
```

Use the directory table above for other categories. `fir_kernel_compare.py`, `bootstrap_stage_a_gate.py`, `raw_se_stage_bc_pareto.py`, and `select_filter.py` analyze the resulting filter-comparison files. Their `--help` documents the comparison-directory and output options.

### Rasch fit and estimator comparison

```bash
mkdir -p results/analysis

# Fit the full random-pool response matrix.
python scripts/analysis/fit_irt.py \
  --data results/constrained_completion/filter_ablation_raw_v2 \
  --output results/analysis/constrained_completion_pool_irt.json

# Compare JML and MML after applying the ALO filter.
python scripts/analysis/jml_vs_mml.py --filter-rule alo \
  --output results/analysis/jml_vs_mml_alo.json
```

For an ALO-filtered bank with prompts and calibration metadata, use `scripts/build_anchor_bundle.py`, as shown below. `fit_irt.py` fits the matrix it is given.

## Constructing a new anchor bank

The downloaded dataset already contains the random pool and calibrated ALO bank. To construct another pool, compose items from seed requests and skills:

```bash
python scripts/random_compose_baseline.py --category constrained_completion \
  --n 500 --seed 42 \
  --output benchmarks/constrained_completion/random_pool_new.jsonl
```

Collect repeated responses from the nine-model panel, then fit the filtered bank:

```bash
python scripts/collect_response_matrix.py --category constrained_completion \
  --dataset benchmarks/constrained_completion/random_pool_new.jsonl \
  --n-trials 40 --output-dir results/constrained_completion/panel_new

python scripts/build_anchor_bundle.py --category constrained_completion \
  --response-dir results/constrained_completion/panel_new \
  --pool benchmarks/constrained_completion/random_pool_new.jsonl \
  --filter-rule alo \
  --output benchmarks/constrained_completion/anchor_new.json
```

The collector runs from the main Python environment and starts each model in its configured vLLM environment. Panel definitions and GPU placement are in `core/panel_config.py`; see [Configuration](../docs/CONFIGURATION.md). The ALO filter retains an item when at least one reference model's success rate lies in `[0.20, 0.80]`.

## Search and adaptive evaluation

| Experiment | Entry point | Configuration or input |
|---|---|---|
| SGBS on one or more targets | `scripts/run_sgbs.py` | `configs/<category>/search_*.yaml`; `--models` selects targets |
| SGBS across reference-panel models | `scripts/run_panel.py` | `--config configs/<category>/search_*.yaml` |
| Adaptive evaluation with augmentation | `main.py` | `configs/<category>/adaptive.yaml` and `--anchor-bundle` |
| SGBS versus uniform selection | `scripts/analysis/sgbs_vs_uniform.py` | Paired `rebuttal_sgbs_k50.yaml` and `rebuttal_uniform_k50.yaml` configurations and their run outputs |

`run_panel.py` runs search for the panel; `collect_response_matrix.py` scores a fixed dataset. Use the collector on accepted SGBS items to obtain the model–item count matrix for boundary-location analysis.

## Paper experiment map

| Experiment | Main scripts | Input artifacts |
|---|---|---|
| Panel leave-one-out stability | `analysis/claim1_calibration_stability.py`, `claim1_loo_tail_diagnostic.py` | Random-pool panel counts |
| Boundary localization | `analysis/claim2_boundary_anchored.py`, `claim2_counterfactual.py`, `claim2_rank_profile.py` | Calibrated anchors and panel counts on SGBS items |
| Adaptive assessment and augmentation | `analysis/claim3_alo_full_vs_untrunc.py`, `claim3_ablation_simulation.py`, `calibrate_claim3_step2.py` | Target responses, CAT replay inputs, and Step-2 outputs |
| Filter selection and information | `analysis/compare_filters.py`, `fir_kernel_compare.py`, `raw_se_stage_bc_pareto.py` | Random-pool counts and filter-comparison JSON |
| Coarse/fine screening | `analysis/empirical_pool_costs_alo.py`, `verify_coarse_fine_fnr.py` | Random-pool counts and analytical screening settings |
| IRT estimator and model-family comparisons | `analysis/jml_vs_mml.py`, `irt_model_compare.py` | Panel and held-out count matrices |
| Truncation and generation budgets | `analysis/truncation_audit.py`, `v_truncation_audit_alo_full.py` | Per-trial responses and completion metadata |
| Compute and token-budget analyses | `analysis/rebuttal_cost_ledger.py`, `rebuttal_n_sensitivity.py`; `rebuttal/test_time_scaling_retrospective.py` | Usage records and stored per-trial responses |
| Human and inter-judge agreement | `rebuttal/run_judge_bias.py`, `analysis/run_judge_pass_rate_diagnostic.py` | Response-level annotations and judge outputs |

Paths in this table are relative to `scripts/`. Some filenames retain experiment identifiers such as `claim1` and `rebuttal` so that configurations and analysis imports continue to match. For a new experiment, record the configuration, random seed, model identifiers, and input paths alongside the results.

## Core settings

| Source | Settings |
|---|---|
| `core/panel_config.py` | Nine reference models, additional target models, per-model launch options |
| `core/inference_budgets.py` | Category and model-specific generation and context limits |
| `configs/<category>/search_*.yaml` | Search selection, composition, trial counts, and boundary intervals |
| `configs/<category>/adaptive.yaml` | CAT, posterior, coverage, and Step-2 settings |

ALO means “at least one” reference model in the interval; AVG filters the mean panel success rate. The ALO bank uses `[0.20, 0.80]`. The canonical adaptive configurations request 50 new items in Step 2. Search trial settings and generation limits should be read from the configuration for the experiment being reproduced.
