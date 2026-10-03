#!/usr/bin/env python3
"""
Convert BFCL Java/JS JSONL eval files into OoO-Spec src_requests format.

Outputs:
  data/train/src_requests/bfcl_dev.jsonl
"""

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing.conversion_utils import (  # noqa: E402
    add_output_arg,
    build_src_request,
    latest_user_content,
    normalize_tool_schema,
    path_arg,
    write_jsonl,
)


def convert_bfcl_row(row: dict, prefix: str, idx: int) -> dict:
    """Convert one BFCL row to src_requests row."""
    functions = row.get("function") or row.get("functions") or []
    if isinstance(functions, dict):
        functions = [functions]
    schemas = [normalize_tool_schema(fn) for fn in functions if isinstance(fn, dict)]
    return build_src_request(
        request_id=f"{prefix}_{idx:05d}",
        source=f"bfcl_{prefix}",
        user_content=latest_user_content(row.get("question", [[]])),
        schemas=schemas,
        answer=None,
        extra_fields={"bfcl_id": row.get("id")},
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=path_arg, default=ROOT / "data" / "evalsets" / "bfcl")
    add_output_arg(parser, ROOT / "data" / "train" / "src_requests" / "bfcl_dev.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    for prefix, filename in [("java", "BFCL_v3_java.json"), ("javascript", "BFCL_v3_javascript.json")]:
        path = args.input_dir / filename
        if not path.exists():
            print(f"Skipping {path}: not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rows.append(convert_bfcl_row(item, prefix, idx))

    write_jsonl(args.output_path, rows)
    print(f"Wrote {len(rows)} BFCL requests to {args.output_path}")


if __name__ == "__main__":
    main()
