#!/usr/bin/env python3
"""Ablation: effect of the three paper-alignment fixes on the dev split.

Compares, on the same samples:
  1. vanilla                       - autoregressive target
  2. toolspec (no history)         - ToolSpec without History Calls retrieval
  3. toolspec (history)            - fix #3: History Calls enabled
  4. ooospec direct sync           - old path: one direct-call sidecar hint, serial
  5. ooospec slots async           - fix #1: paper-style parallel slot wave
  6. ooospec slots async (history) - fix #1+#2+#3: parallel slots, within-request
                                     asynchronous non-blocking pickup, History Calls

Usage:
    python src/inference/run_ablation_proposer.py [N]
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "toolspec"))

import torch  # noqa: E402

from inference.ooospec_pipeline import (  # noqa: E402
    generate_sidecar_hint,
    get_paths,
    load_sidecar_tokenizer,
    load_target,
    load_vanilla_target,
    run_toolspec_sample,
    run_vanilla_sample,
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
DEV = ROOT / "data" / "train" / "src_requests" / "dev.jsonl"
rows = [json.loads(line) for line in open(DEV)][:N]


def gold_norm(row):
    gold = row.get("answer")
    while isinstance(gold, list) and gold:
        gold = gold[0]
    if isinstance(gold, dict):
        return gold
    if isinstance(gold, str):
        try:
            return json.loads(gold[gold.find("{"):gold.rfind("}") + 1])
        except Exception:  # noqa: BLE001
            return None
    return None


def score(text, gold):
    try:
        obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception:  # noqa: BLE001
        return False, False
    if not isinstance(obj, dict) or not gold:
        return False, False
    name_ok = obj.get("name") == gold.get("name")
    params_ok = name_ok and (
        json.dumps(obj.get("parameters", {}), sort_keys=True, ensure_ascii=False)
        == json.dumps(gold.get("parameters", {}), sort_keys=True, ensure_ascii=False)
    )
    return name_ok, params_ok


def main():
    paths = get_paths()
    sidecar_tok = load_sidecar_tokenizer(paths["sidecar_base"])
    target_model, target_tok = load_target(paths["target_model"], "cuda:0")
    vanilla = load_vanilla_target(paths["target_model"], "cuda:2")
    adj = torch.zeros((target_model.vocab_size, 8), dtype=torch.long, device=target_model.device)

    def run_config(name, fn):
        times, names, params, steps, accepts, sidecars, waits = [], 0, 0, [], [], [], []
        for row in rows:
            res = fn(row)
            times.append(res["wall_time"])
            steps.append(res.get("decoding_steps", 0))
            accepts.append(res.get("mean_accept", 0.0))
            sidecars.append(res.get("sidecar_time", 0.0))
            waits.append(res.get("sidecar_wait", 0.0))
            n_ok, p_ok = score(res["target_output"], gold_norm(row))
            names += int(n_ok)
            params += int(p_ok)
        k = len(rows)
        print(f"{name:32s} name={names/k:.3f} params={params/k:.3f} "
              f"t_avg={statistics.mean(times):.3f}s t_p50={statistics.median(times):.3f}s "
              f"steps={statistics.mean(steps):.2f} accept={statistics.mean(accepts):.2f} "
              f"sidecar={statistics.mean(sidecars):.3f}s wait={statistics.mean(waits):.3f}s")

    run_config("vanilla", lambda r: run_vanilla_sample(r, vanilla, target_tok))
    run_config("toolspec (no history)",
               lambda r: run_toolspec_sample(r, sidecar_tok, target_model, target_tok, adj,
                                             use_hint=False))

    mem_t = []
    run_config("toolspec (history)",
               lambda r: run_toolspec_sample(r, sidecar_tok, target_model, target_tok, adj,
                                             use_hint=False, output_memory=mem_t))

    def ooo_direct(row):
        raw, sidecar_time = generate_sidecar_hint(sidecar_tok, row)
        return run_toolspec_sample(row, sidecar_tok, target_model, target_tok, adj,
                                   use_hint=True, prefetched_hint=(raw, sidecar_time))

    run_config("ooospec direct sync (no hist)", ooo_direct)
    run_config("ooospec slots async (no hist)",
               lambda r: run_toolspec_sample(r, sidecar_tok, target_model, target_tok, adj,
                                             use_hint=True, slot_mode=True))

    mem_o = []
    run_config("ooospec slots async (history)",
               lambda r: run_toolspec_sample(r, sidecar_tok, target_model, target_tok, adj,
                                             use_hint=True, slot_mode=True, output_memory=mem_o))


if __name__ == "__main__":
    main()
