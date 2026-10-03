#!/usr/bin/env python3
"""Convert BFCL v3 'simple' category from SimpleTool parquet to src_requests format."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.conversion_utils import (  # noqa: E402
    add_output_arg,
    build_src_request,
    normalize_tool_schema,
    path_arg,
    write_jsonl,
)


def parse_ground_truth(gt_raw) -> dict | None:
    if not gt_raw or gt_raw == [{}]:
        return None
    gt = gt_raw[0] if isinstance(gt_raw, list) else gt_raw
    if not isinstance(gt, dict):
        return None
    func_name = list(gt.keys())[0]
    args = gt[func_name]
    if isinstance(args, dict):
        # BFCL ground_truth may wrap scalar args in single-element lists.
        params = {}
        for k, v in args.items():
            if isinstance(v, list) and len(v) == 1:
                params[k] = v[0]
            else:
                params[k] = v
        return {"name": func_name, "parameters": params}
    return {"name": func_name, "parameters": {}}


def convert_row(row, idx: int) -> dict | None:
    turns = json.loads(row["turns"])
    query = turns[0][-1]["content"] if turns and turns[0] and turns[0][-1] else ""
    tools = json.loads(row["tools"])
    schemas = [normalize_tool_schema(tool) for tool in tools]

    gt = parse_ground_truth(json.loads(row["ground_truth"]))
    if gt is None:
        return None

    return build_src_request(
        request_id=f"bfcl_v3_simple_{idx:05d}",
        source="bfcl_v3_simple",
        user_content=query,
        schemas=schemas,
        answer=gt,
        extra_fields={"bfcl_id": row.get("id")},
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=path_arg,
        default=ROOT / "data" / "evalsets" / "bfcl_v3_train.parquet",
    )
    add_output_arg(parser, ROOT / "data" / "train" / "src_requests" / "bfcl_v3_simple.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.input_path)
    df = df[df["test_category"] == "simple"].reset_index(drop=True)

    rows = []
    for idx, row in df.iterrows():
        converted = convert_row(row, idx)
        if converted is not None:
            rows.append(converted)

    write_jsonl(args.output_path, rows)
    print(f"Wrote {len(rows)} BFCL v3 simple requests to {args.output_path}")


if __name__ == "__main__":
    main()
