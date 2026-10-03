#!/usr/bin/env python3
"""
Offline (L2) evaluation of the LoRA sidecar on semantic rows.

Computes:
  - function-index accuracy
  - argument-value exact-match
  - null precision / recall
  - direct-call exact-match, function-name accuracy, argument F1
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_paths():
    root = Path(__file__).resolve().parents[2]
    return {
        "model_name": os.environ.get("OOOSPEC_SIDECAR_BASE", "Qwen/Qwen3-0.6B"),
        "adapter_path": root / "outputs" / "sidecar-lora" / "final",
        "dev_rows": root / "data" / "train" / "rows" / "dev.jsonl",
        "output_dir": root / "outputs" / "sidecar-eval",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=get_paths()["model_name"])
    parser.add_argument("--adapter-path", default=str(get_paths()["adapter_path"]))
    parser.add_argument("--dev-rows", default=str(get_paths()["dev_rows"]))
    parser.add_argument("--output-dir", default=str(get_paths()["output_dir"]))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=31082027)
    return parser.parse_args()


def load_rows(path: Path):
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_direct_call(text: str):
    """Parse a direct-call JSON output. Return (name, params) or None."""
    text = text.strip()
    if not text:
        return None
    # Try to extract the first {...} block if the model added chatter.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    params = obj.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
    return name, params


def arg_f1(pred_params: dict, gold_params: dict):
    """Token-level F1 over JSON-stringified argument values."""
    pred_set = set()
    gold_set = set()
    for k, v in pred_params.items():
        pred_set.add((k, compact_json(v)))
    for k, v in gold_params.items():
        gold_set.add((k, compact_json(v)))
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(pred_set & gold_set)
    return 2 * tp / (len(pred_set) + len(gold_set))


def evaluate(predictions, rows):
    metrics = {
        "function_index": {"n": 0, "correct": 0},
        "argument_value": {"n": 0, "correct": 0, "null_tp": 0, "null_pred": 0, "null_gold": 0},
        "direct_call": {"n": 0, "exact": 0, "name_correct": 0, "arg_f1_sum": 0.0},
    }

    for pred, row in zip(predictions, rows):
        gold = row["messages"][-1]["content"].strip()
        pred_text = pred.strip()
        row_type = row["row_type"]

        if row_type == "function_index":
            metrics["function_index"]["n"] += 1
            try:
                if int(pred_text) == int(gold):
                    metrics["function_index"]["correct"] += 1
            except ValueError:
                pass

        elif row_type == "argument_value":
            metrics["argument_value"]["n"] += 1
            if pred_text == gold:
                metrics["argument_value"]["correct"] += 1
            gold_null = gold == "null"
            pred_null = pred_text == "null"
            if gold_null:
                metrics["argument_value"]["null_gold"] += 1
            if pred_null:
                metrics["argument_value"]["null_pred"] += 1
            if gold_null and pred_null:
                metrics["argument_value"]["null_tp"] += 1

        elif row_type == "direct_call":
            metrics["direct_call"]["n"] += 1
            gold_call = parse_direct_call(gold)
            pred_call = parse_direct_call(pred_text)
            if gold_call and pred_call:
                g_name, g_params = gold_call
                p_name, p_params = pred_call
                if compact_json({"name": g_name, "parameters": g_params}) == compact_json(
                    {"name": p_name, "parameters": p_params}
                ):
                    metrics["direct_call"]["exact"] += 1
                if g_name == p_name:
                    metrics["direct_call"]["name_correct"] += 1
                metrics["direct_call"]["arg_f1_sum"] += arg_f1(p_params, g_params)

    results = {}
    fi = metrics["function_index"]
    results["function_index_accuracy"] = fi["correct"] / fi["n"] if fi["n"] else 0.0

    av = metrics["argument_value"]
    results["argument_exact_match"] = av["correct"] / av["n"] if av["n"] else 0.0
    results["null_precision"] = av["null_tp"] / av["null_pred"] if av["null_pred"] else 0.0
    results["null_recall"] = av["null_tp"] / av["null_gold"] if av["null_gold"] else 0.0

    dc = metrics["direct_call"]
    results["direct_call_exact_match"] = dc["exact"] / dc["n"] if dc["n"] else 0.0
    results["direct_call_name_accuracy"] = dc["name_correct"] / dc["n"] if dc["n"] else 0.0
    results["direct_call_argument_f1"] = dc["arg_f1_sum"] / dc["n"] if dc["n"] else 0.0

    results["counts"] = metrics
    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    rows = load_rows(Path(args.dev_rows))
    print(f"Dev rows: {len(rows)}")

    # Build prompts
    prompts = []
    for row in rows:
        prompt_text = tokenizer.apply_chat_template(
            row["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt_text)

    # Batch tokenize & generate
    all_preds = []
    device = next(model.parameters()).device
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decode only generated tokens
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_preds.extend(decoded)

    # Save predictions
    pred_path = output_dir / "dev_predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for row, pred in zip(rows, all_preds):
            rec = {
                "request_id": row.get("request_id"),
                "row_type": row["row_type"],
                "function_index": row.get("function_index"),
                "param_name": row.get("param_name"),
                "gold": row["messages"][-1]["content"],
                "pred": pred,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Predictions saved to {pred_path}")

    results = evaluate(all_preds, rows)
    counts = results.pop("counts")

    print("\n=== Counts ===")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print("\n=== Metrics ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "counts": counts}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
