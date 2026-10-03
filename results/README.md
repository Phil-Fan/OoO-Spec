# Results

Aggregate evaluation results for this **independent reproduction** of OoO-Spec.
Only aggregate metrics are published here — no per-request model outputs, no
dataset text, and no internal ablation experiments.

## Setup

| Item | Value |
|---|---|
| Target | Qwen2.5-7B-Instruct, FP16, greedy, batch size 1 |
| Sidecar | Qwen3-0.6B + one LoRA adapter (frozen), served with vLLM |
| Placement | target and sidecar on two separate GPUs (split configuration) |
| Hardware | 4 × NVIDIA A100-PCIE-40GB |
| Methods | Vanilla AR, ToolSpec, OoO-Spec |

## Files

| File | Description |
|---|---|
| `all_benchmarks_summary.json` | Vanilla / ToolSpec / OoO-Spec summary across five benchmarks |
| `sidecar_l2_metrics.json` | Offline (no target) quality of the LoRA sidecar on semantic rows |
| `dev_529_summary.json` | Dev split (API-Bank + ToolAlpaca, n=529) comparison |
| `bfcl_java_js_n150_summary.json` | BFCL Java/JS (n=150) comparison |

## Main results (OoO-Spec vs Vanilla)

| Benchmark | n | Vanilla (s) | ToolSpec (s) | OoO-Spec (s) | ToolSpec / Vanilla | OoO / Vanilla | OoO / ToolSpec |
|---|---:|---:|---:|---:|---:|---:|---:|
| BFCL v3 simple | 400 | 1.121 | 0.492 | 0.259 | 2.28× | 4.33× | 1.90× |
| BFCL v4 simple | 550 | 1.171 | 0.416 | 0.288 | 2.81× | 4.07× | 1.45× |
| Mobile Actions | 961 | 1.626 | 0.749 | 0.471 | 2.17× | 3.45× | 1.59× |
| SealTools | 1354 | 1.733 | 0.780 | 0.634 | 2.22× | 2.73× | 1.23× |
| OpenFunctions v1 | 112 | 0.967 | 0.353 | 0.267 | 2.74× | 3.62× | 1.32× |
| BFCL Java/JS | 150 | 1.427 | 0.546 | 0.432 | 2.61× | 3.30× | 1.26× |
| Dev (API-Bank + ToolAlpaca) | 529 | 1.075 | 0.446 | 0.335 | 2.41× | 3.21× | 1.33× |

## Sidecar offline quality (`sidecar_l2_metrics.json`)

| Metric | Value |
|---|---:|
| Function-index accuracy | 96.3% |
| Argument exact match | 86.1% |
| Null precision / recall | 95.6% / 96.8% |
| Direct-call exact match | 54.2% |
| Direct-call function-name accuracy | 95.5% |
| Direct-call argument F1 | 0.657 |

## Caveats

- These numbers were produced by an independent re-implementation and are the
  reproducer's own measurements, not the authors' official run.
- The reported accuracy of `bfcl_v3_simple` (name 57.0%, params 17.5%) looks
  anomalously low compared with `bfcl_v4_simple` (99.6% / 94.9%) and is most
  likely a gold-answer parsing/format mismatch rather than a model failure. The
  latency numbers are unaffected.
- Internal ablations and all per-request result files are intentionally not
  published.

## Regenerating

The aggregation scripts live in `src/inference/` (e.g.
`run_all_benchmarks_eval.py`). Data and model weights are not committed; build
them from `data/download_datasets.py` and `src/training/train_sidecar_lora.py`.
