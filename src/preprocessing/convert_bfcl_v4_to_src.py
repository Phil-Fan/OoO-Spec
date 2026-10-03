#!/usr/bin/env python3
"""Convert BFCL v4 simple eval files to src_requests format."""

import argparse
import json
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


def convert_row(row: dict, prefix: str, idx: int) -> dict:
    functions = row.get("function") or row.get("functions") or []
    if isinstance(functions, dict):
        functions = [functions]
    schemas = [normalize_tool_schema(fn) for fn in functions if isinstance(fn, dict)]
    return build_src_request(
        request_id=f"bfcl_v4_{prefix}_{idx:05d}",
        source=f"bfcl_v4_{prefix}",
        user_content=latest_user_content(row.get("question", [[]])),
        schemas=schemas,
        answer=None,
        extra_fields={"bfcl_id": row.get("id")},
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=path_arg, default=ROOT / "data" / "evalsets" / "bfcl_v4")
    add_output_arg(parser, ROOT / "data" / "train" / "src_requests" / "bfcl_v4_simple.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    for prefix, filename in [
        ("java", "BFCL_v4_simple_java.json"),
        ("javascript", "BFCL_v4_simple_javascript.json"),
        ("python", "BFCL_v4_simple_python.json"),
    ]:
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
                rows.append(convert_row(item, prefix, idx))

    write_jsonl(args.output_path, rows)
    print(f"Wrote {len(rows)} BFCL v4 simple requests to {args.output_path}")


if __name__ == "__main__":
    main()
