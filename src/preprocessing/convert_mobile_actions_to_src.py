#!/usr/bin/env python3
"""Convert Mobile Actions eval split to src_requests format (single-turn)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.conversion_utils import (  # noqa: E402
    add_output_arg,
    build_src_request,
    normalize_tool_schema,
    path_arg,
    write_jsonl,
)


def convert_sample(sample: dict, idx: int) -> dict | None:
    if sample.get("metadata") != "eval":
        return None

    tools = sample.get("tools", [])
    schemas = [normalize_tool_schema(t) for t in tools if isinstance(t, dict)]

    messages = sample.get("messages", [])
    # Find the first user message and the immediately following assistant tool_calls.
    user_content = ""
    gold_call = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            # Look ahead for assistant tool_calls
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "assistant":
                    tool_calls = messages[j].get("tool_calls", [])
                    if tool_calls:
                        tc = tool_calls[0].get("function", tool_calls[0])
                        gold_call = {
                            "name": tc.get("name", ""),
                            "parameters": tc.get("arguments", {}),
                        }
                    break
            break

    if not user_content or gold_call is None:
        return None

    return build_src_request(
        request_id=f"mobile_actions_{idx:05d}",
        source="mobile_actions",
        user_content=user_content,
        schemas=schemas,
        answer=gold_call,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=path_arg,
        default=ROOT / "data" / "evalsets" / "mobile_actions_dataset.jsonl",
    )
    add_output_arg(parser, ROOT / "data" / "train" / "src_requests" / "mobile_actions_eval.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            converted = convert_sample(sample, idx)
            if converted is not None:
                rows.append(converted)

    write_jsonl(args.output_path, rows)
    print(f"Wrote {len(rows)} mobile actions eval requests to {args.output_path}")


if __name__ == "__main__":
    main()
