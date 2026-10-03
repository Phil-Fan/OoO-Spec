#!/usr/bin/env python3
"""
Retry failed teacher traces with a stronger prompt.

Reads existing teacher_traces/{split}.jsonl, finds entries where call is None,
regenerates them with an appended instruction forcing a tool call, and updates
 the file in place.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(ROOT))

from preprocessing.generate_teacher_traces_vllm import extract_tool_calls, prompt_hash


def get_paths():
    return {
        "api_base": "http://127.0.0.1:8000/v1",
        "model": os.environ.get("OOOSPEC_TEACHER_MODEL", "Qwen/Qwen2.5-32B-Instruct"),
        "train_traces": ROOT / "data" / "train" / "teacher_traces" / "train.jsonl",
        "dev_traces": ROOT / "data" / "train" / "teacher_traces" / "dev.jsonl",
    }


RETRY_SUFFIX = (
    "\n\nImportant: You must make exactly one tool call. "
    "Output ONLY a JSON object in the format: "
    '{"name": "FunctionName", "parameters": {"arg1": "value1", ...}}'
)


async def generate_one(
    client: httpx.AsyncClient,
    api_base: str,
    model: str,
    row: dict,
) -> dict:
    messages = row.get("messages", [])
    # Append force-call instruction to the last user message
    messages = [dict(m) for m in messages]
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = messages[-1]["content"] + RETRY_SUFFIX

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


async def retry_failed(
    trace_path: Path,
    src_path: Path,
    api_base: str,
    model: str,
    concurrency: int = 16,
):
    trace_path = Path(trace_path)
    src_path = Path(src_path)

    if not trace_path.exists():
        print(f"Trace file not found: {trace_path}")
        return

    traces = [json.loads(line) for line in open(trace_path, "r", encoding="utf-8")]
    src_rows = {r["request_id"]: r for r in (json.loads(line) for line in open(src_path, "r", encoding="utf-8"))}

    failed_ids = [t["request_id"] for t in traces if t.get("call") is None]
    print(f"Found {len(failed_ids)} failed traces in {trace_path}")
    if not failed_ids:
        return

    failed_src_rows = [src_rows[r_id] for r_id in failed_ids if r_id in src_rows]

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency)) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_retry(row):
            async with semaphore:
                try:
                    return await generate_one(client, api_base, model, row)
                except Exception as e:
                    return {
                        "request_id": row["request_id"],
                        "prompt_hash": prompt_hash(row["system"], row["user"]),
                        "call": None,
                        "raw_output": "",
                        "error": str(e),
                    }

        tasks = [bounded_retry(row) for row in failed_src_rows]
        retry_results = {}
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            retry_results[result["request_id"]] = result
            if (i + 1) % 50 == 0:
                print(f"  retried {i + 1}/{len(failed_src_rows)}")

    # Update traces in place
    updated = 0
    new_success = 0
    for t in traces:
        if t.get("call") is None and t["request_id"] in retry_results:
            new_result = retry_results[t["request_id"]]
            # Preserve original raw_output for debugging if still failing
            if new_result.get("call") is not None:
                t["call"] = new_result["call"]
                t["raw_output"] = new_result["raw_output"]
                t.pop("error", None)
                new_success += 1
            else:
                t["raw_output"] = new_result["raw_output"]
                t["retry_error"] = new_result.get("error", "still no call")
            updated += 1

    # Sort by request_id and save
    traces.sort(key=lambda r: r["request_id"])
    with open(trace_path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Updated {updated} entries, {new_success} newly successful")
    print(f"Saved to {trace_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=get_paths()["api_base"])
    parser.add_argument("--model", default=get_paths()["model"])
    parser.add_argument("--trace-path")
    parser.add_argument("--src-path")
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    if args.trace_path and args.src_path:
        trace_path = Path(args.trace_path)
        src_path = Path(args.src_path)
    else:
        trace_path = get_paths()[f"{args.split}_traces"]
        src_path = ROOT / "data" / "train" / "src_requests" / f"{args.split}.jsonl"

    asyncio.run(
        retry_failed(trace_path, src_path, args.api_base, args.model, args.concurrency)
    )


if __name__ == "__main__":
    main()
