#!/usr/bin/env python3
"""
Generate teacher traces for OoO-Spec source requests.

Teacher: Qwen2.5-32B-Instruct
Decoding: greedy, batch size 1

Input:  data/train/src_requests/{train,dev}.jsonl
Output: data/train/teacher_traces/{train,dev}.jsonl

Each output line:
  {"request_id": "...", "prompt_hash": "...", "call": {"name": "...", "parameters": {...}}}
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def get_paths():
    return {
        "model_path": os.environ.get("OOOSPEC_TEACHER_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
        "train_input": ROOT / "data" / "train" / "src_requests" / "train.jsonl",
        "dev_input": ROOT / "data" / "train" / "src_requests" / "dev.jsonl",
        "train_output": ROOT / "data" / "train" / "teacher_traces" / "train.jsonl",
        "dev_output": ROOT / "data" / "train" / "teacher_traces" / "dev.jsonl",
    }


def prompt_hash(system: str, user: str) -> str:
    text = system + "\n" + user
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_tool_calls(text: str) -> list[dict]:
    """
    Extract tool calls from model output.
    Supports formats:
      <tool_call>{"name": "...", "parameters": {...}}</tool_call>
      {"name": "...", "parameters": {...}}
    Returns list of {"name": str, "parameters": dict}.
    """
    calls = []

    # Try <tool_call> blocks first
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        inner = match.group(1).strip()
        # Might contain one or more JSON objects
        for obj_match in re.finditer(r"\{.*?\}", inner, re.DOTALL):
            try:
                obj = json.loads(obj_match.group(0))
                if "name" in obj and "parameters" in obj:
                    calls.append({"name": obj["name"], "parameters": obj["parameters"]})
            except json.JSONDecodeError:
                continue

    # Fallback: search for standalone JSON objects with name+parameters
    if not calls:
        for match in re.finditer(r"\{.*?\"name\".*?\"parameters\".*?\}", text, re.DOTALL):
            try:
                obj = json.loads(match.group(0))
                if "name" in obj and "parameters" in obj:
                    calls.append({"name": obj["name"], "parameters": obj["parameters"]})
            except json.JSONDecodeError:
                continue

    return calls


def generate_traces(
    input_path: Path,
    output_path: Path,
    model_path: str,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    dtype: str = "bfloat16",
    device_map: str = "auto",
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(input_path, "r", encoding="utf-8")]
    print(f"Loaded {len(rows)} requests from {input_path}")

    # Lazy import so the script can be inspected without torch installed
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    results = []
    generated = 0
    failed = 0

    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            prompts = []
            for row in batch:
                messages = row.get("messages", [])
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for row, output_ids in zip(batch, outputs):
                input_len = inputs["input_ids"].shape[1]
                new_ids = output_ids[input_len:]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                calls = extract_tool_calls(text)

                if calls:
                    first_call = calls[0]
                    result = {
                        "request_id": row["request_id"],
                        "prompt_hash": prompt_hash(row["system"], row["user"]),
                        "call": first_call,
                        "raw_output": text,
                    }
                    generated += 1
                else:
                    result = {
                        "request_id": row["request_id"],
                        "prompt_hash": prompt_hash(row["system"], row["user"]),
                        "call": None,
                        "raw_output": text,
                    }
                    failed += 1

                results.append(result)

            if (i // batch_size + 1) % 10 == 0:
                print(f"  processed {min(i + batch_size, len(rows))}/{len(rows)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done: generated={generated}, failed={failed}, total={len(rows)}")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=get_paths()["model_path"])
    parser.add_argument("--train-input", default=str(get_paths()["train_input"]))
    parser.add_argument("--dev-input", default=str(get_paths()["dev_input"]))
    parser.add_argument("--train-output", default=str(get_paths()["train_output"]))
    parser.add_argument("--dev-output", default=str(get_paths()["dev_output"]))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--split", choices=["train", "dev", "both"], default="both")
    args = parser.parse_args()

    if args.split in ("train", "both"):
        generate_traces(
            Path(args.train_input),
            Path(args.train_output),
            args.model_path,
            args.batch_size,
            args.max_new_tokens,
            args.dtype,
            args.device_map,
        )

    if args.split in ("dev", "both"):
        generate_traces(
            Path(args.dev_input),
            Path(args.dev_output),
            args.model_path,
            args.batch_size,
            args.max_new_tokens,
            args.dtype,
            args.device_map,
        )


if __name__ == "__main__":
    main()
