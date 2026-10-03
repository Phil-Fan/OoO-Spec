#!/usr/bin/env python3
"""
Expand teacher traces into semantic rows for sidecar LoRA training.

Inputs:
  - data/train/src_requests/{train,dev}.jsonl
  - data/train/teacher_traces/{train,dev}.jsonl

Outputs:
  - data/train/rows/{train,dev}.jsonl

Each source request with a valid teacher call expands into:
  1. One function-index row
  2. One argument-value-or-null row per (function, parameter) in schemas
  3. One direct-call row (training only)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def get_paths():
    return {
        "train_src": ROOT / "data" / "train" / "src_requests" / "train.jsonl",
        "dev_src": ROOT / "data" / "train" / "src_requests" / "dev.jsonl",
        "train_traces": ROOT / "data" / "train" / "teacher_traces" / "train.jsonl",
        "dev_traces": ROOT / "data" / "train" / "teacher_traces" / "dev.jsonl",
        "train_output": ROOT / "data" / "train" / "rows" / "train.jsonl",
        "dev_output": ROOT / "data" / "train" / "rows" / "dev.jsonl",
    }


SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer the user's request using the available tools."
)


def format_schemas_with_indices(schemas: list[dict]) -> str:
    lines = []
    for idx, s in enumerate(schemas):
        params = s.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        param_list = ", ".join(params.keys()) if params else ""
        lines.append(f"[{idx}] {s['name']}({param_list})")
    return "\n".join(lines)


def format_schemas_with_param_tags(schemas: list[dict]) -> str:
    lines = []
    for idx, s in enumerate(schemas):
        params = s.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        for p_name in params:
            lines.append(f"[{idx}] {s['name']} param: {p_name}")
    return "\n".join(lines)


def compact_json(value) -> str:
    """Return compact JSON string without extra spaces."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def spaced_json(value) -> str:
    """Return JSON string with spaces, matching the target model's output format."""
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def expand_request(src_row: dict, trace: dict) -> list[dict]:
    """Expand one source request + teacher trace into semantic rows."""
    schemas = src_row.get("schemas", [])
    call = trace.get("call")
    if not call:
        return []

    selected_name = call.get("name")
    selected_params = call.get("parameters", {})

    # Find selected function index
    try:
        selected_idx = next(i for i, s in enumerate(schemas) if s["name"] == selected_name)
    except StopIteration:
        # Teacher call references a function not in schemas; skip
        return []

    dialogue = src_row.get("user", "")
    # Strip the trailing instruction to keep only the dialogue history
    dialogue = dialogue.replace(
        "\n\n<user> Based on our conversation above, please only make one tool call to solve my need.</user>",
        "",
    )

    rows = []
    global_idx = len(rows)

    # 1. Function-index row
    user_prompt = (
        f"{dialogue}\n\n"
        "Available tools:\n"
        f"{format_schemas_with_indices(schemas)}\n\n"
        "Which tool should be called? Respond with only the index number."
    )
    rows.append(
        {
            "request_id": src_row["request_id"],
            "row_type": "function_index",
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": str(selected_idx)},
            ],
        }
    )

    # 2. Argument-value-or-null rows for every (function, parameter)
    for f_idx, schema in enumerate(schemas):
        fn_name = schema["name"]
        params = schema.get("parameters", {})
        for p_name in params:
            user_prompt = (
                f"{dialogue}\n\n"
                "Available tool parameters:\n"
                f"{format_schemas_with_param_tags(schemas)}\n\n"
                f"What is the value for parameter '{p_name}' of tool [{f_idx}] {fn_name}? "
                "Respond with a compact JSON value, or the literal string null if not applicable."
            )
            if f_idx == selected_idx and p_name in selected_params:
                target = compact_json(selected_params[p_name])
            else:
                target = "null"
            rows.append(
                {
                    "request_id": src_row["request_id"],
                    "row_type": "argument_value",
                    "function_index": f_idx,
                    "param_name": p_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": target},
                    ],
                }
            )

    # 3. Direct-call row (training auxiliary)
    user_prompt = (
        f"{dialogue}\n\n"
        "Available tools:\n"
        f"{format_schemas_with_indices(schemas)}\n\n"
        "What is the complete tool call? Respond with a JSON object."
    )
    rows.append(
        {
            "request_id": src_row["request_id"],
            "row_type": "direct_call",
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": spaced_json(call)},
            ],
        }
    )

    return rows


def expand_split(src_path: Path, traces_path: Path, output_path: Path):
    src_rows = {r["request_id"]: r for r in (json.loads(l) for l in open(src_path, "r", encoding="utf-8"))}
    traces = [json.loads(l) for l in open(traces_path, "r", encoding="utf-8")]

    all_rows = []
    skipped = 0
    for trace in traces:
        req_id = trace["request_id"]
        if req_id not in src_rows:
            skipped += 1
            continue
        if trace.get("call") is None:
            skipped += 1
            continue
        rows = expand_request(src_rows[req_id], trace)
        all_rows.extend(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Expanded {len(traces) - skipped} requests -> {len(all_rows)} rows")
    print(f"Skipped {skipped} requests (missing src or no valid call)")
    print(f"Saved to {output_path}")

    # Count row types
    type_counts = {}
    for row in all_rows:
        t = row["row_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("Row type counts:", type_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-src", default=str(get_paths()["train_src"]))
    parser.add_argument("--dev-src", default=str(get_paths()["dev_src"]))
    parser.add_argument("--train-traces", default=str(get_paths()["train_traces"]))
    parser.add_argument("--dev-traces", default=str(get_paths()["dev_traces"]))
    parser.add_argument("--train-output", default=str(get_paths()["train_output"]))
    parser.add_argument("--dev-output", default=str(get_paths()["dev_output"]))
    parser.add_argument("--split", choices=["train", "dev", "both"], default="both")
    args = parser.parse_args()

    if args.split in ("train", "both"):
        print("\n=== Train ===")
        expand_split(Path(args.train_src), Path(args.train_traces), Path(args.train_output))

    if args.split in ("dev", "both"):
        print("\n=== Dev ===")
        expand_split(Path(args.dev_src), Path(args.dev_traces), Path(args.dev_output))


if __name__ == "__main__":
    main()
