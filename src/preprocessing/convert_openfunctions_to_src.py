#!/usr/bin/env python3
"""Convert Gorilla OpenFunctions v1 test set to src_requests format."""

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


def parse_python_call(s: str):
    """Parse 'name(arg1=val1, arg2=val2)' into (name, {arg:val})."""
    s = s.strip()
    m = re.match(r"([A-Za-z_][A-Za-z0-9_\.]*)\s*\((.*)\)\s*$", s, re.DOTALL)
    if not m:
        return None
    name, args_str = m.group(1), m.group(2).strip()
    args = {}
    if args_str:
        # Split by top-level commas.
        depth = 0
        cur = ""
        parts = []
        for ch in args_str:
            if ch in "({[":
                depth += 1
            elif ch in ")}]":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        parts.append(cur.strip())
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    v = ast.literal_eval(v)
                except Exception:
                    v = v.strip('"').strip("'")
                args[k] = v
    return name, args


def convert_function(fn: dict) -> dict:
    return normalize_tool_schema(fn, name_key="api_call")


def convert_sample(sample: dict, idx: int) -> dict | None:
    question = sample.get("question", "")
    fn = sample.get("function", {})
    model_answer = sample.get("model_answer", "")

    parsed = parse_python_call(model_answer)
    if parsed is None:
        return None
    name, params = parsed

    schema = convert_function(fn)
    return build_src_request(
        request_id=f"openfunctions_{idx:05d}",
        source="openfunctions",
        user_content=question,
        schemas=[schema],
        answer={"name": name, "parameters": params},
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=path_arg,
        default=ROOT / "data" / "evalsets" / "openfunctions" / "gorilla_openfunctions_v1_test.json",
    )
    add_output_arg(parser, ROOT / "data" / "train" / "src_requests" / "openfunctions_eval.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for idx, sample in enumerate(data):
        converted = convert_sample(sample, idx)
        if converted is not None:
            rows.append(converted)

    write_jsonl(args.output_path, rows)
    print(f"Wrote {len(rows)} OpenFunctions eval requests to {args.output_path}")


if __name__ == "__main__":
    main()
