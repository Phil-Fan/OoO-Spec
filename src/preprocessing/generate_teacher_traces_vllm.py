#!/usr/bin/env python3
"""
Generate teacher traces using a vLLM-served Qwen2.5-32B-Instruct.

Assumes vLLM OpenAI-compatible API is running at http://127.0.0.1:8000/v1.

Input:  data/train/src_requests/{train,dev}.jsonl
Output: data/train/teacher_traces/{train,dev}.jsonl

Output format per line:
  {"request_id": "...", "prompt_hash": "...", "call": {"name": "...", "parameters": {...}}}
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def get_paths():
    return {
        "api_base": "http://127.0.0.1:8000/v1",
        "model": os.environ.get("OOOSPEC_TEACHER_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
        "train_input": ROOT / "data" / "train" / "src_requests" / "train.jsonl",
        "dev_input": ROOT / "data" / "train" / "src_requests" / "dev.jsonl",
        "train_output": ROOT / "data" / "train" / "teacher_traces" / "train.jsonl",
        "dev_output": ROOT / "data" / "train" / "teacher_traces" / "dev.jsonl",
    }


def prompt_hash(system: str, user: str) -> str:
    text = system + "\n" + user
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_json_objects(text: str) -> list[dict]:
    """Extract all balanced {...} blocks and return those with name+parameters."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                candidate = text[i:j]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
            i = j
        else:
            i += 1
    return results


def extract_tool_calls(text: str) -> list[dict]:
    """Extract {"name":..., "parameters":...} from assistant output."""
    calls = []

    # 1. Try markdown code blocks (```json ... ```)
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL):
        inner = match.group(1).strip()
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
                calls.append({"name": obj["name"], "parameters": obj["parameters"]})
                continue
        except json.JSONDecodeError:
            pass
        calls.extend(extract_json_objects(inner))

    # 2. Try <tool_call> blocks
    if not calls:
        for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
            calls.extend(extract_json_objects(match.group(1)))

    # 3. Fallback: scan entire text for name+parameters JSON objects
    if not calls:
        calls.extend(extract_json_objects(text))

    # 4. Last fallback: API-Bank style [Name(params)]
    if not calls:
        import ast

        def extract_bracketed(t):
            results = []
            i = 0
            while i < len(t):
                if t[i] == "[":
                    depth = 1
                    j = i + 1
                    while j < len(t) and depth > 0:
                        if t[j] == "[":
                            depth += 1
                        elif t[j] == "]":
                            depth -= 1
                        j += 1
                    if depth == 0:
                        results.append(t[i + 1 : j - 1])
                    i = j
                else:
                    i += 1
            return results

        for inner in extract_bracketed(text):
            m = re.match(r"(\w+)\((.*)\)$", inner.strip(), re.DOTALL)
            if not m:
                continue
            name, args_str = m.group(1), m.group(2)
            params = {}
            if args_str.strip():
                for seg in args_str.split(","):
                    kv = seg.split("=", 1)
                    if len(kv) == 2:
                        v = kv[1].strip()
                        if (v.startswith("'") and v.endswith("'")) or (
                            v.startswith('"') and v.endswith('"')
                        ):
                            v = v[1:-1]
                        try:
                            v = ast.literal_eval(v)
                        except Exception:
                            pass
                        params[kv[0].strip()] = v
            calls.append({"name": name, "parameters": params})

    return calls


async def generate_one(client: httpx.AsyncClient, api_base: str, model: str, row: dict) -> dict:
    messages = row.get("messages", [])
    response = await client.post(
        f"{api_base}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1024,
            "top_p": 1.0,
            "n": 1,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"].get("content", "")
    calls = extract_tool_calls(text)
    return {
        "request_id": row["request_id"],
        "prompt_hash": prompt_hash(row["system"], row["user"]),
        "call": calls[0] if calls else None,
        "raw_output": text,
    }


async def generate_traces(
    input_path: Path,
    output_path: Path,
    api_base: str,
    model: str,
    concurrency: int = 16,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(input_path, "r", encoding="utf-8")]
    print(f"Loaded {len(rows)} requests from {input_path}")

    results = []
    generated = 0
    failed = 0

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency)) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_generate(row):
            nonlocal generated, failed
            async with semaphore:
                try:
                    result = await generate_one(client, api_base, model, row)
                    if result["call"] is not None:
                        generated += 1
                    else:
                        failed += 1
                    return result
                except Exception as e:
                    failed += 1
                    return {
                        "request_id": row["request_id"],
                        "prompt_hash": prompt_hash(row["system"], row["user"]),
                        "call": None,
                        "raw_output": "",
                        "error": str(e),
                    }

        tasks = [bounded_generate(row) for row in rows]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if (i + 1) % 100 == 0:
                print(f"  processed {i + 1}/{len(rows)} (ok={generated}, fail={failed})")

    # Sort by request_id for determinism
    results.sort(key=lambda r: r["request_id"])

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done: generated={generated}, failed={failed}, total={len(rows)}")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=get_paths()["api_base"])
    parser.add_argument("--model", default=get_paths()["model"])
    parser.add_argument("--train-input", default=str(get_paths()["train_input"]))
    parser.add_argument("--dev-input", default=str(get_paths()["dev_input"]))
    parser.add_argument("--train-output", default=str(get_paths()["train_output"]))
    parser.add_argument("--dev-output", default=str(get_paths()["dev_output"]))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--split", choices=["train", "dev", "both"], default="both")
    args = parser.parse_args()

    if args.split in ("train", "both"):
        asyncio.run(
            generate_traces(
                Path(args.train_input),
                Path(args.train_output),
                args.api_base,
                args.model,
                args.concurrency,
            )
        )

    if args.split in ("dev", "both"):
        asyncio.run(
            generate_traces(
                Path(args.dev_input),
                Path(args.dev_output),
                args.api_base,
                args.model,
                args.concurrency,
            )
        )


if __name__ == "__main__":
    main()
