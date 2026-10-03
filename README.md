# OoO-Spec — Out-of-Order Semantic Speculation for Fast Tool Calling

> **This repository is an independent reproduction, not the authors' official release.**
>
> It re-implements the method described in *OoO-Spec: Out-of-Order Semantic
> Speculation for Fast Tool Calling* (arXiv:2608.00814). The code, data
> pipeline, sidecar training, and online inference loop here were rebuilt from
> the paper and its supplementary material. It is shared for reproducibility and
> research; it is not endorsed by or affiliated with the original authors.
>
> Reproduced by Phil-Fan (2026). The experimental results reported from this
> code base are the reproducer's own measurements, produced independently from
> the authors' official run.

## Status

- ✅ Data pipeline (source requests → teacher traces → semantic rows)
- ✅ Qwen3-0.6B + LoRA sidecar training
- ✅ Split-GPU asynchronous inference loop on top of ToolSpec
- 📊 Aggregate metrics are published under [`results/`](results/).
  Per-request outputs, raw datasets, model weights, and internal ablations are
  not committed.

If you use this reproduction, please cite the original paper (see
[Citation](#citation)) and note that the numbers you obtain come from an
independent re-implementation.

## Repository layout

```text
src/
  preprocessing/     data pipeline (downloads → src_requests → traces → rows)
  training/          LoRA sidecar training (train_sidecar_lora.py)
  inference/         OoO-Spec pipeline and benchmark evaluations
  toolspec/          vendored ToolSpec modules (Apache-2.0, third-party)
data/
  README.md          data contract and pipeline description (Chinese)
  download_datasets.py   downloads public datasets into data/raw, data/evalsets
results/             aggregate evaluation metrics
thesis/              paper source (internal, not tracked)
plan/                internal reproduction plan (not tracked)
outputs/             generated artifacts (not tracked)
```

`data/raw`, `data/evalsets`, `data/train`, `outputs/`, `output/`,
`notion_report.md`, and `plan/` are gitignored on purpose: they contain
third-party or derived data, private notes, and large artifacts that should not
be redistributed in the repository.

## Method in one paragraph

A frozen **Qwen3-0.6B + LoRA sidecar** predicts the function choice and every
schema-defined argument slot for a request **in one parallel wave**, out of
textual order. The target model runs a native **ToolSpec** decoding loop and,
without ever blocking, joins the sidecar's rendered semantic hint at recurring
candidate-construction boundaries. The target remains the only verifier and
commit authority. One sidecar is reused across target sizes and families.

## Requirements

- Python 3.12
- One GPU is enough to run the offline pipeline; the online system expects two
  GPUs (target + sidecar), which is the main configuration.
- The target side and the sidecar side use different dependency sets:
  - **target / ToolSpec**: `torch==2.5.1`, `transformers==4.51.1`
  - **sidecar**: served with **vLLM** (OpenAI-compatible API)
  - Two virtual environments are recommended, matching `src/toolspec/requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, if all you need is to run the vendored ToolSpec backend, use its
own pins:

```bash
pip install -r src/toolspec/requirements.txt
```

## Pipeline

All commands assume the repository root as the working directory and write to
the default `data/` and `outputs/` locations. Most scripts also accept
`OOOSPEC_*` environment variables for paths (see
[Environment variables](#environment-variables)).

### 1. Download public datasets

```bash
python data/download_datasets.py
```

This populates `data/raw/` and `data/evalsets/`. Some upstream datasets do not
have a stable public URL; pass them explicitly if needed:

```bash
python data/download_datasets.py \
  --mobile-actions-url URL \
  --bfcl-v3-simple-url URL
```

### 2. Build source requests (train / dev split)

```bash
python src/preprocessing/prepare_src_requests.py
```

Produces `data/train/src_requests/train.jsonl` and `dev.jsonl` (targets:
9,662 / 529). Contamination filtering removes any API whose name or function
name appears in the evaluation inventory.

### 3. Generate teacher traces

Either use a local GPU with HF Transformers:

```bash
python src/preprocessing/generate_teacher_traces.py \
  --model-path Qwen/Qwen2.5-32B-Instruct
```

or a running vLLM OpenAI-compatible server (default
`http://127.0.0.1:8000/v1`):

```bash
python src/preprocessing/generate_teacher_traces_vllm.py \
  --api-base http://127.0.0.1:8000/v1
```

Optionally retry traces that failed to parse:

```bash
python src/preprocessing/retry_failed_traces.py --split dev
```

Produces `data/train/teacher_traces/{train,dev}.jsonl`.

### 4. Expand into semantic rows

```bash
python src/preprocessing/expand_semantic_rows.py
```

Produces `data/train/rows/{train,dev}.jsonl` (function-index,
argument-value-or-null, and auxiliary direct-call rows).

### 5. Train the LoRA sidecar

```bash
python src/training/train_sidecar_lora.py \
  --model-name Qwen/Qwen3-0.6B \
  --output-dir outputs/sidecar-lora
```

LoRA `r=32, alpha=64, dropout=0.05`; 1 epoch; effective batch 32; cosine LR
`1e-4`; prompt labels masked. The adapter is written to
`outputs/sidecar-lora/final`.

### 6. Convert evaluation benchmarks (optional)

Each converter writes a unified `src_requests` JSONL:

```bash
python src/preprocessing/convert_bfcl_to_src.py
python src/preprocessing/convert_bfcl_v3_simple_to_src.py
python src/preprocessing/convert_bfcl_v4_to_src.py
python src/preprocessing/convert_mobile_actions_to_src.py
python src/preprocessing/convert_sealtools_to_src.py
python src/preprocessing/convert_openfunctions_to_src.py
```

Run any of them with `--help` for input/output overrides.

### 7. Serve the sidecar

Start a vLLM server with the LoRA adapter enabled:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --enable-lora \
  --lora-modules sidecar=outputs/sidecar-lora/final \
  --max-lora-rank 32 \
  --port 7892 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 2048
```

The inference scripts send requests to `http://localhost:7892/v1/completions`
by default.

### 8. Run evaluations

Offline sidecar quality (no target model needed):

```bash
python src/inference/eval_sidecar_lora.py \
  --adapter-path outputs/sidecar-lora/final
```

Single dev-set comparison of Vanilla / ToolSpec / OoO-Spec:

```bash
python src/inference/run_ooospec_eval.py
```

Combined evaluation across all configured benchmarks:

```bash
python src/inference/run_all_benchmarks_eval.py
```

The scripts place the target on `cuda:0` and the Vanilla reference on `cuda:2`
by default, matching the split-GPU configuration. Results are written under
`outputs/`.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `OOOSPEC_RAW_DATA_DIR` | root of downloaded raw data | `data/raw` |
| `OOOSPEC_TEACHER_MODEL` | teacher model id/path | `Qwen/Qwen2.5-32B-Instruct` |
| `OOOSPEC_SIDECAR_BASE` | sidecar base model | `Qwen/Qwen3-0.6B` |
| `OOOSPEC_TARGET_MODEL` | target model id/path | `Qwen/Qwen2.5-7B-Instruct` |
| `OOOSPEC_APIBANK_LV1_TRAIN` / `LV2` / `LV3` | API-Bank train files | `data/raw/api_bank/...` |
| `OOOSPEC_TOOLSPEC_APIBANK_DIR` | processed API-Bank eval dir | `data/evalsets/apibank` |

Never commit credentials. `.env` files are gitignored.

## Third-party components and data

- **ToolSpec** is vendored under `src/toolspec/` and is licensed
  **Apache-2.0**, Copyright Heming Xia et al. See `src/toolspec/PROVENANCE.md`
  and `src/toolspec/LICENSE`. It is not part of this reproduction's original
  work.
- Evaluation datasets (API-Bank, ToolAlpaca, BFCL, Mobile Actions, SealTools,
  OpenFunctions) and the teacher/target models (Qwen2.5, Qwen3) are the property
  of their respective authors and are **not redistributed** here. Use the
  download scripts and respect each upstream license.
- See `NOTICE` for the full attribution list.

## Citation

If you use this reproduction, please cite the original method paper:

```bibtex
@misc{zhang2026ooospec,
  title        = {OoO-Spec: Out-of-Order Semantic Speculation for Fast Tool Calling},
  author       = {Zhiheng Zhang and Mujie Xu and Feiyu Sun and Zhixin Zhang},
  year         = {2026},
  eprint       = {2608.00814},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2608.00814}
}
```

and the baseline this builds on:

```bibtex
@article{xia2026toolspec,
  title   = {ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding},
  author  = {Xia, Heming and Li, Yongqi and Du, Cunxiao and Song, Mingbo and Li, Wenjie},
  journal = {arXiv preprint arXiv:2604.13519},
  year    = {2026}
}
```

## License

This reproduction is released under the **Apache License 2.0** (see `LICENSE`).
Vendored third-party code keeps its own license; see `NOTICE` and
`src/toolspec/LICENSE`.
