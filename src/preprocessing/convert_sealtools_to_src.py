#!/usr/bin/env python3
"""Convert SealTools test sets (with api_list) to src_requests format."""

import argparse
import ast
import json
import re
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


def extract_api_list_and_instruction(text: str):
    """Extract api_list JSON and task_instruction from the prompt text."""
    m = re.search(r"api_list\s*=\s*(\[.*?\])\s*\ntask_instruction\s*=\s*\"(.*?)\"\s*\nOutput:", text, re.DOTALL)
    if not m:
        return None, None
    api_list_str, instruction = m.group(1), m.group(2)
    try:
        api_list = ast.literal_eval(api_list_str)
    except Exception:
        return None, None
    return api_list, instruction


def parse_gold_answer(text: str):
    """Parse gold answer from gpt response like [{'api': '...', 'parameters': {...}}]."""
    text = text.strip()
    try:
        obj = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(obj, list) and obj:
        call = obj[0]
        return {"name": call.get("api", ""), "parameters": call.get("parameters", {})}
    return None


def convert_api(api: dict) -> dict:
    return normalize_tool_schema(
        api,
        name_key="api_name",
        description_key="api_description",
    )


def convert_sample(sample: dict, prefix: str, idx: int) -> dict | None:
    conversations = sample.get("conversations", [])
    if len(conversations) < 2:
        return None
    human_text = conversations[0].get("value", "")
    gpt_text = conversations[1].get("value", "")

    api_list, instruction = extract_api_list_and_instruction(human_text)
    if api_list is None or instruction is None:
        return None

    gold = parse_gold_answer(gpt_text)
    if gold is None:
        return None

    schemas = [convert_api(api) for api in api_list]
    return build_src_request(
        request_id=f"sealtools_{prefix}_{idx:05d}",
        source=f"sealtools_{prefix}",
        user_content=instruction,
        schemas=schemas,
        answer=gold,
        extra_fields={"sealtools_id": sample.get("id")},
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=path_arg,
        default=ROOT / "data" / "evalsets" / "sealtools",
    )
    add_output_arg(parser, ROOT / "data" / "train" / "src_requests" / "sealtools_eval.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    for prefix, filename in [
        ("in_domain", "test_in_domain.json"),
        ("out_domain", "test_out_domain.json"),
    ]:
        path = args.input_dir / filename
        if not path.exists():
            print(f"Skipping {path}: not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, sample in enumerate(data):
            converted = convert_sample(sample, prefix, idx)
            if converted is not None:
                rows.append(converted)

    write_jsonl(args.output_path, rows)
    print(f"Wrote {len(rows)} SealTools eval requests to {args.output_path}")


if __name__ == "__main__":
    main()
