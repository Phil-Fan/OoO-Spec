#!/usr/bin/env python3
"""
Full dev-set OoO-Spec evaluation.

Runs Vanilla, ToolSpec, and OoO-Spec on all src_requests/dev rows,
records outputs, latency, and correctness metrics vs gold answers.
"""

import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT / "toolspec"))

from inference.ooospec_pipeline import (  # noqa: E402
    SIDECAR_API_URL,
    SidecarPrefetcher,
    build_sidecar_prompt_text,
    decode_target_output,
    generate_sidecar_hint,
    get_paths,
    load_sidecar_tokenizer,
    load_src_requests,
    load_target,
    load_vanilla_target,
    parse_sidecar_output,
    run_toolspec_sample,
    run_vanilla_sample,
)


def parse_call_json(text: str):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj.get("name"), obj.get("parameters", {})


def evaluate_output(pred_text: str, gold):
    name_correct = False
    params_exact = False
    pred = parse_call_json(pred_text)
    if pred is None:
        return {"name_correct": False, "params_exact": False}
    p_name, p_params = pred
    g_name = gold.get("name") if isinstance(gold, dict) else gold[0].get("name")
    g_params = gold.get("parameters", {}) if isinstance(gold, dict) else gold[0].get("parameters", {})
    name_correct = p_name == g_name
    if name_correct:
        # Exact match after compact JSON normalization.
        params_exact = (
            json.dumps(p_params, ensure_ascii=False, sort_keys=True)
            == json.dumps(g_params, ensure_ascii=False, sort_keys=True)
        )
    return {"name_correct": name_correct, "params_exact": params_exact}


def main():
    paths = get_paths()
    src_rows = load_src_requests(paths["src_dev"])

    target_device = "cuda:0"
    vanilla_device = "cuda:2"

    sidecar_tok = load_sidecar_tokenizer(paths["sidecar_base"])
    target_model, target_tok = load_target(paths["target_model"], target_device)
    vanilla_model = load_vanilla_target(paths["target_model"], vanilla_device)

    adj_matrix_hint = torch.zeros(
        (target_model.vocab_size, 8),
        dtype=torch.long,
        device=target_model.device,
        requires_grad=False,
    )
    adj_matrix_baseline = torch.zeros_like(adj_matrix_hint)

    output_dir = ROOT / "outputs" / "ooospec_eval_prefetch"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "dev_results.jsonl"
    summary_path = output_dir / "summary.json"

    # Resume from existing partial results if any.
    processed_ids = set()
    records = []
    if result_path.exists():
        for line in open(result_path, "r", encoding="utf-8"):
            rec = json.loads(line)
            records.append(rec)
            processed_ids.add(rec["request_id"])
        print(f"Resuming: {len(records)} records already processed.")

    src_rows = [r for r in src_rows if r.get("request_id") not in processed_ids]

    prefetcher = SidecarPrefetcher(SIDECAR_API_URL)
    try:
        # Prime the first hint so sample 0 has something to wait on.
        if src_rows:
            first_row = src_rows[0]
            prefetcher.submit(
                first_row.get("request_id"),
                build_sidecar_prompt_text(sidecar_tok, first_row),
            )

        for i, src_row in enumerate(tqdm(src_rows, desc="Evaluating dev", initial=len(records), total=len(src_rows) + len(records))):
            request_id = src_row.get("request_id")
            gold = src_row.get("answer")
            if isinstance(gold, list) and gold:
                gold = gold[0]

            # Wait for the already-submitted sidecar hint.
            hint_wait_start = time.time()
            raw_hint, sidecar_time = prefetcher.get(request_id)
            sidecar_wait = time.time() - hint_wait_start

            # Immediately schedule the sidecar for the next sample so it runs
            # concurrently with vanilla / ToolSpec / OoO target on the current sample.
            if i + 1 < len(src_rows):
                next_row = src_rows[i + 1]
                prefetcher.submit(
                    next_row.get("request_id"),
                    build_sidecar_prompt_text(sidecar_tok, next_row),
                )

            vanilla_res = run_vanilla_sample(src_row, vanilla_model, target_tok)
            toolspec_res = run_toolspec_sample(
                src_row,
                sidecar_tok,
                target_model,
                target_tok,
                adj_matrix_baseline,
                use_hint=False,
            )
            ooospec_res = run_toolspec_sample(
                src_row,
                sidecar_tok,
                target_model,
                target_tok,
                adj_matrix_hint,
                use_hint=True,
                prefetched_hint=(raw_hint, sidecar_time),
            )

            # Total OoO latency = target time + any residual wait for the prefetched hint.
            ooospec_total = ooospec_res["wall_time"] + sidecar_wait

            record = {
                "request_id": request_id,
                "gold": gold,
                "vanilla": {
                    "output": vanilla_res["target_output"],
                    "time": vanilla_res["wall_time"],
                    "new_tokens": vanilla_res["new_tokens"],
                    **evaluate_output(vanilla_res["target_output"], gold),
                },
                "toolspec": {
                    "output": toolspec_res["target_output"],
                    "time": toolspec_res["wall_time"],
                    "new_tokens": toolspec_res["new_tokens"],
                    "decoding_steps": toolspec_res["decoding_steps"],
                    "mean_accept": toolspec_res["mean_accept"],
                    **evaluate_output(toolspec_res["target_output"], gold),
                },
                "ooospec": {
                    "output": ooospec_res["target_output"],
                    "time": ooospec_res["wall_time"],
                    "sidecar_time": sidecar_time,
                    "sidecar_wait": sidecar_wait,
                    "total_time": ooospec_total,
                    "new_tokens": ooospec_res["new_tokens"],
                    "decoding_steps": ooospec_res["decoding_steps"],
                    "mean_accept": ooospec_res["mean_accept"],
                    "sidecar_raw": ooospec_res["sidecar_raw"],
                    "sidecar_formatted": ooospec_res["sidecar_formatted"],
                    **evaluate_output(ooospec_res["target_output"], gold),
                },
            }
            records.append(record)
            with open(result_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        prefetcher.close()

    # Aggregate summary
    def avg(key, subkey):
        return sum(r[subkey][key] for r in records) / len(records)

    def acc(subkey, key):
        return sum(1 for r in records if r[subkey][key]) / len(records)

    summary = {
        "n": len(records),
        "vanilla": {
            "avg_time": avg("time", "vanilla"),
            "avg_new_tokens": avg("new_tokens", "vanilla"),
            "name_acc": acc("vanilla", "name_correct"),
            "params_exact_acc": acc("vanilla", "params_exact"),
        },
        "toolspec": {
            "avg_time": avg("time", "toolspec"),
            "avg_new_tokens": avg("new_tokens", "toolspec"),
            "avg_steps": avg("decoding_steps", "toolspec"),
            "avg_mean_accept": avg("mean_accept", "toolspec"),
            "name_acc": acc("toolspec", "name_correct"),
            "params_exact_acc": acc("toolspec", "params_exact"),
        },
        "ooospec": {
            "avg_time": avg("time", "ooospec"),
            "avg_total_time": avg("total_time", "ooospec"),
            "avg_sidecar_time": avg("sidecar_time", "ooospec"),
            "avg_sidecar_wait": avg("sidecar_wait", "ooospec"),
            "avg_new_tokens": avg("new_tokens", "ooospec"),
            "avg_steps": avg("decoding_steps", "ooospec"),
            "avg_mean_accept": avg("mean_accept", "ooospec"),
            "name_acc": acc("ooospec", "name_correct"),
            "params_exact_acc": acc("ooospec", "params_exact"),
        },
        "speedups": {
            "toolspec_vs_vanilla": avg("time", "vanilla") / avg("time", "toolspec"),
            "ooospec_target_vs_vanilla": avg("time", "vanilla") / avg("time", "ooospec"),
            "ooospec_total_vs_vanilla": avg("time", "vanilla") / avg("total_time", "ooospec"),
            "ooospec_vs_toolspec": avg("time", "toolspec") / avg("time", "ooospec"),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== Summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetailed results: {result_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
