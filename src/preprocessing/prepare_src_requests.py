#!/usr/bin/env python3
"""
Prepare OoO-Spec source requests from raw datasets.

Inputs:
  - API-Bank train JSON files
  - HuggingFace cache: Ahren09/ToolAlpaca (train split)
  - Eval sets used for contamination filtering

Outputs:
  - data/train/src_requests/train.jsonl (9,662 rows target)
  - data/train/src_requests/dev.jsonl   (529 rows target)

Processing steps:
  1. Load raw API-Bank lv1 train and ToolAlpaca train.
  2. Convert both to a unified schema + dialogue format.
  3. Build contamination list from eval sets (API/function names).
  4. Remove any training API whose name or function name appears in eval.
  5. Deterministic prompt-hash split into train/dev.
  6. Save JSONL and write a manifest.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Allow importing from the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def get_paths():
    raw_data_dir = env_path("OOOSPEC_RAW_DATA_DIR", ROOT / "data" / "raw")
    return {
        "apibank_train": env_path(
            "OOOSPEC_APIBANK_LV1_TRAIN",
            raw_data_dir / "api_bank" / "lv1-train.json",
        ),
        "apibank_train_lv2": env_path(
            "OOOSPEC_APIBANK_LV2_TRAIN",
            raw_data_dir / "api_bank" / "lv2-train.json",
        ),
        "apibank_train_lv3": env_path(
            "OOOSPEC_APIBANK_LV3_TRAIN",
            raw_data_dir / "api_bank" / "lv3-train.json",
        ),
        "toolalpaca_hf": "Ahren09/ToolAlpaca",
        "toolspec_apibank_dir": env_path(
            "OOOSPEC_TOOLSPEC_APIBANK_DIR",
            ROOT / "data" / "evalsets" / "apibank",
        ),
        "bfcl_v3_java": env_path(
            "OOOSPEC_BFCL_V3_JAVA",
            ROOT / "data" / "evalsets" / "bfcl" / "BFCL_v3_java.json",
        ),
        "bfcl_v3_javascript": env_path(
            "OOOSPEC_BFCL_V3_JAVASCRIPT",
            ROOT / "data" / "evalsets" / "bfcl" / "BFCL_v3_javascript.json",
        ),
        "bfcl_v4_java": env_path(
            "OOOSPEC_BFCL_V4_JAVA",
            ROOT / "data" / "evalsets" / "bfcl_v4" / "BFCL_v4_simple_java.json",
        ),
        "bfcl_v4_javascript": env_path(
            "OOOSPEC_BFCL_V4_JAVASCRIPT",
            ROOT / "data" / "evalsets" / "bfcl_v4" / "BFCL_v4_simple_javascript.json",
        ),
        "output_train": ROOT / "data" / "train" / "src_requests" / "train.jsonl",
        "output_dev": ROOT / "data" / "train" / "src_requests" / "dev.jsonl",
        "manifest": ROOT / "data" / "train" / "src_requests" / "manifest.json",
    }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def extract_bracketed(text: str) -> list[str]:
    """Extract top-level [ ... ] blocks, handling nested brackets."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == "[":
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                j += 1
            if depth == 0:
                results.append(text[i + 1 : j - 1])
            i = j
        else:
            i += 1
    return results


def split_apibank_args(args_str: str) -> list[str]:
    """Split API-Bank args string into top-level key=value segments."""
    segments = []
    i = 0
    start = 0
    in_quote = False
    quote_char = None
    bracket_depth = 0
    while i < len(args_str):
        c = args_str[i]
        if not in_quote and c in "'\"":
            in_quote = True
            quote_char = c
        elif in_quote and c == quote_char:
            if i > 0 and args_str[i - 1] != "\\":
                in_quote = False
                quote_char = None
        elif not in_quote:
            if c in "[{":
                bracket_depth += 1
            elif c in "]}":
                bracket_depth -= 1
            elif c == "," and bracket_depth == 0:
                segments.append(args_str[start:i].strip())
                start = i + 1
        i += 1
    if start < len(args_str):
        segments.append(args_str[start:].strip())
    return segments


def parse_apibank_arg_value(raw: str) -> Any:
    """Parse a single API-Bank argument value."""
    raw = raw.strip()
    # Strip matching outer quotes
    if (raw.startswith("'") and raw.endswith("'")) or (
        raw.startswith('"') and raw.endswith('"')
    ):
        raw = raw[1:-1]
    # Try JSON first (handles double-quoted dicts/lists)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try Python literal (handles single-quoted structures)
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    return raw


def parse_apibank_output(output: str):
    """
    Parse API-Bank output like:
      API-Request: [Get_All_Sessions()]
      API-Request: [ModifyRegistration(appointment_id='34567890', ...)]
    Returns list of {"name": str, "parameters": dict}.
    Robust to nested lists/dicts and imperfect quoting in the dataset.
    """
    calls = []
    for inner in extract_bracketed(output):
        inner = inner.strip()
        if not inner:
            continue
        m = re.match(r"(\w+)\((.*)\)$", inner, re.DOTALL)
        if not m:
            continue
        name, args_str = m.group(1), m.group(2)
        params = {}
        if args_str.strip():
            for seg in split_apibank_args(args_str):
                kv = seg.split("=", 1)
                if len(kv) == 2:
                    params[kv[0].strip()] = parse_apibank_arg_value(kv[1].strip())
        calls.append({"name": name, "parameters": params})
    return calls


def parse_apibank_input(input_text: str):
    """
    API-Bank input contains multiple JSON API descriptions followed by user text.
    Returns (schemas, user_text).
    """
    schemas = []
    lines = input_text.strip().splitlines()
    consumed = 0
    for line in lines:
        line = line.strip()
        if not line:
            consumed += 1
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and "apiCode" in obj:
            schemas.append(
                {
                    "name": obj["apiCode"],
                    "description": obj.get("description", ""),
                    "parameters": obj.get("parameters", {}),
                }
            )
            consumed += 1
        else:
            break

    user_text = "\n".join(lines[consumed:]).strip()
    # Remove "User: " prefix if present
    if user_text.startswith("User:"):
        user_text = user_text[len("User:") :].strip()
    return schemas, user_text


def sanitize_schema_value(value):
    """Recursively make schema parameter values JSON-serializable."""
    if isinstance(value, set):
        return sorted(value) if all(isinstance(x, (int, float, str)) for x in value) else list(value)
    if isinstance(value, (list, tuple)):
        return [sanitize_schema_value(v) for v in value]
    if isinstance(value, dict):
        return {k: sanitize_schema_value(v) for k, v in value.items()}
    return value


def parse_pipe_tool_result(result: str) -> dict | None:
    """
    Parse pipe-delimited ToolSearcher result used in API-Bank lv2:
      API: tool_name | Description: ... | Input parameters: {...} | Output result: ...
    Returns a schema dict {"name", "description", "parameters"}.
    """
    result = result.strip()
    if (result.startswith('"') and result.endswith('"')) or (
        result.startswith("'") and result.endswith("'")
    ):
        result = result[1:-1]

    name_match = re.search(r"API:\s*(\w+)", result)
    if not name_match:
        return None
    name = name_match.group(1).strip()

    desc_match = re.search(r"Description:\s*(.*?)\s*\|\s*Input parameters:", result)
    description = desc_match.group(1).strip() if desc_match else ""

    params = {}
    params_match = re.search(
        r"Input parameters:\s*(\{.*?\})\s*\|\s*Output result:", result, re.DOTALL
    )
    if not params_match:
        params_match = re.search(r"Input parameters:\s*(\{.*?\})\s*$", result.strip(), re.DOTALL)
    if params_match:
        params_str = params_match.group(1).strip()
        try:
            params = ast.literal_eval(params_str)
        except Exception:
            try:
                params = json.loads(params_str.replace("'", '"'))
            except Exception:
                params = {}

    return {
        "name": name,
        "description": description,
        "parameters": sanitize_schema_value(params),
    }


def parse_json_tool_result(result: str) -> dict | None:
    """
    Parse JSON-style ToolSearcher result used in API-Bank lv3:
      {'name': '...', 'description': '...', 'input_parameters': {...}, 'output_parameters': {...}}
    Returns a schema dict {"name", "description", "parameters"}.
    """
    result = result.strip()
    if (result.startswith('"') and result.endswith('"')) or (
        result.startswith("'") and result.endswith("'")
    ):
        result = result[1:-1]

    try:
        obj = ast.literal_eval(result)
    except Exception:
        try:
            obj = json.loads(result.replace("'", '"'))
        except Exception:
            return None

    if isinstance(obj, list):
        obj = obj[0] if obj else None
    if not isinstance(obj, dict):
        return None

    params = obj.get("input_parameters") or obj.get("parameters") or {}
    return {
        "name": obj.get("name", ""),
        "description": obj.get("description", ""),
        "parameters": sanitize_schema_value(params),
    }


def extract_tool_searcher_results(input_text: str) -> list[dict]:
    """
    Scan API-Bank dialogue history for ToolSearcher results and extract the
    retrieved concrete tool schemas. Deduplicates by tool name.
    """
    tools = []
    seen_names = set()

    for line in input_text.split("\n"):
        line = line.strip()
        if "ToolSearcher" not in line or "]->" not in line:
            continue

        idx = line.find("]->")
        result = line[idx + 3 :].strip()
        if (result.startswith('"') and result.endswith('"')) or (
            result.startswith("'") and result.endswith("'")
        ):
            result = result[1:-1]

        if not result:
            continue

        tool = None
        if result.startswith("{"):
            tool = parse_json_tool_result(result)
        elif result.startswith("API:") or result.startswith("API "):
            tool = parse_pipe_tool_result(result)

        if tool and tool.get("name") and tool["name"] not in seen_names:
            tools.append(tool)
            seen_names.add(tool["name"])

    return tools


def resolve_apibank_schemas(schemas: list[dict], calls: list[dict], input_text: str) -> list[dict] | None:
    """
    For API-Bank lv2/lv3 the input only declares ToolSearcher, but the actual
    call is a concrete tool retrieved by ToolSearcher in the dialogue history.
    Extract those retrieved tools and use them as the effective schemas.

    Returns None if the called tool cannot be resolved against available schemas.
    """
    if not schemas or not calls:
        return None

    first_call = calls[0]
    # lv1: concrete schemas already, keep as-is
    if not (len(schemas) == 1 and schemas[0].get("name") == "ToolSearcher"):
        return schemas

    # First turn of lv2/lv3: output calls ToolSearcher, schemas are correct
    if first_call["name"] == "ToolSearcher":
        return schemas

    # Second+ turn: output calls concrete tool. Extract retrieved tools.
    retrieved = extract_tool_searcher_results(input_text)
    retrieved_names = {t["name"] for t in retrieved}
    if first_call["name"] in retrieved_names:
        return retrieved

    # Cannot resolve: called tool not found in ToolSearcher results
    return None


def build_system_prompt_apibank(schemas: list[dict]) -> str:
    """Build a concise system prompt from API-Bank schemas."""
    tools_text = []
    for idx, s in enumerate(schemas, start=1):
        params = s.get("parameters", {})
        params_repr = json.dumps(params, ensure_ascii=False)
        tools_text.append(
            f"{idx}. Name: {s['name']}\n"
            f"Description: {s.get('description', '')}\n"
            f"Parameters: {params_repr}"
        )
    tools_block = "\n".join(tools_text)
    return (
        "You are a helpful assistant that selects the best available tool and calls it.\n\n"
        "**Available Tools**\n"
        f"{tools_block}\n\n"
        "**Output Format**\n"
        "Output ONLY a single JSON object with exactly two keys:\n"
        '  "name": the name of the tool to call\n'
        '  "parameters": a JSON object of parameter names to values\n'
        "Do not wrap the JSON in markdown, code fences, or any explanation."
    )


def parse_toolalpaca_documentation(nl_documentation: str) -> list[dict]:
    """
    Parse ToolAlpaca nl_documentation into list of tool schemas.
    Format example:
      sendHttpRequest: Send an HTTP request...
      Parameters: {"method": "...", "url": "..."}
      Output: ...
    """
    tools = []
    lines = nl_documentation.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if (
            line
            and ":" in line
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("Parameters:")
            and not line.startswith(("Output:", "-", "Parameters:"))
        ):
            name, description = line.split(":", 1)
            params_text = lines[i + 1].strip()[len("Parameters:") :].strip()
            try:
                raw_params = json.loads(params_text)
            except json.JSONDecodeError:
                raw_params = {}

            params = {}
            for param_name, param_desc in raw_params.items():
                if isinstance(param_desc, dict):
                    params[param_name] = param_desc
                else:
                    params[param_name] = {
                        "type": "str",
                        "description": str(param_desc),
                    }

            tools.append(
                {
                    "name": name.strip(),
                    "description": description.strip(),
                    "parameters": params,
                }
            )
        i += 1
    return tools


def parse_toolalpaca_golden(golden_answer: list[dict]) -> list[dict]:
    """Convert ToolAlpaca golden_answer to unified call format."""
    calls = []
    for item in golden_answer:
        if not isinstance(item, dict):
            continue
        action = item.get("Action", "")
        action_input = item.get("Action_Input", "{}")
        try:
            params = json.loads(action_input) if isinstance(action_input, str) else action_input
        except json.JSONDecodeError:
            params = {}
        calls.append({"name": action, "parameters": params})
    return calls


def build_system_prompt_toolalpaca(tools: list[dict]) -> str:
    """Build a concise system prompt from ToolAlpaca tool schemas."""
    tools_text = []
    for idx, tool in enumerate(tools, start=1):
        params = tool.get("parameters", {})
        params_repr = json.dumps(params, ensure_ascii=False)
        tools_text.append(
            f"{idx}. Name: {tool['name']}\n"
            f"Description: {tool.get('description', '')}\n"
            f"Parameters: {params_repr}"
        )
    tools_block = "\n".join(tools_text)
    return (
        "You are a helpful assistant that selects the best available tool and calls it.\n\n"
        "**Available Tools**\n"
        f"{tools_block}\n\n"
        "**Output Format**\n"
        "Output ONLY a single JSON object with exactly two keys:\n"
        '  "name": the name of the tool to call\n'
        '  "parameters": a JSON object of parameter names to values\n'
        "Do not wrap the JSON in markdown, code fences, or any explanation."
    )


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_apibank_train(paths: list[Path]):
    """Load and convert API-Bank train files (lv1, lv2, lv3)."""
    requests = []
    total_skipped = 0
    global_idx = 0
    for path in paths:
        if not path.exists():
            print(f"API-Bank file not found, skipping: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        skipped = 0
        for row in data:
            raw_schemas, user_text = parse_apibank_input(row.get("input", ""))
            if not raw_schemas:
                skipped += 1
                continue
            calls = parse_apibank_output(row.get("output", ""))
            if not calls:
                skipped += 1
                continue

            schemas = resolve_apibank_schemas(raw_schemas, calls, row.get("input", ""))
            if schemas is None:
                skipped += 1
                continue

            system = build_system_prompt_apibank(schemas)
            user = (
                "**Dialogue Records History**\n"
                f"<user>{user_text}</user>\n\n"
                "<user> Based on our conversation above, please only make one tool call to solve my need.</user>"
            )

            requests.append(
                {
                    "request_id": f"apibank_{global_idx:05d}",
                    "source": "apibank",
                    "system": system,
                    "user": user,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "schemas": schemas,
                    "answer": calls,
                    "api_name": None,
                    "function_names": [s["name"] for s in schemas],
                }
            )
            global_idx += 1
        total_skipped += skipped
        print(f"API-Bank {path.name}: processed {len(data)} rows, skipped {skipped}")
    print(f"API-Bank total: loaded {len(requests)} requests, skipped {total_skipped}")
    return requests


def load_toolalpaca_train(repo_id: str):
    """Load and convert ToolAlpaca train split from HuggingFace cache."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' library is required. Activate the ToolSpec venv or install it."
        ) from e

    ds = load_dataset(repo_id)
    requests = []
    skipped = 0
    for idx, row in enumerate(ds["train"]):
        tools = parse_toolalpaca_documentation(row.get("nl_documentation", ""))
        if not tools:
            skipped += 1
            continue
        calls = parse_toolalpaca_golden(row.get("golden_answer", []))
        if not calls:
            skipped += 1
            continue

        system = build_system_prompt_toolalpaca(tools)
        instruction = row.get("instruction", "").strip()
        user = (
            "**Dialogue Records History**\n"
            f"<user>{instruction}</user>\n\n"
            "<user> Based on our conversation above, please only make one tool call to solve my need.</user>"
        )

        requests.append(
            {
                "request_id": f"toolalpaca_{idx:05d}",
                "source": "toolalpaca",
                "system": system,
                "user": user,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "schemas": tools,
                "answer": calls,
                "api_name": row.get("name"),
                "function_names": [t["name"] for t in tools],
            }
        )
    print(f"ToolAlpaca: loaded {len(requests)} requests, skipped {skipped}")
    return requests


# ---------------------------------------------------------------------------
# Decontamination
# ---------------------------------------------------------------------------


def extract_names_from_system_prompt(system: str) -> set[str]:
    """Extract tool/API/function names from a ToolSpec-style system prompt."""
    names = set()
    for m in re.finditer(r"Name:\s*(\w+)", system):
        names.add(m.group(1).strip())
    return names


def extract_names_from_bfcl(path: Path) -> set[str]:
    """Extract function names from BFCL JSONL eval file."""
    names = set()
    if not path.exists():
        return names
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            # BFCL format: function list under "function" or "functions"
            functions = item.get("function") or item.get("functions") or []
            if isinstance(functions, dict):
                functions = [functions]
            for fn in functions:
                if isinstance(fn, dict):
                    name = fn.get("name") or fn.get("api_name")
                    if name:
                        names.add(str(name))
    return names


def build_eval_name_set(
    toolspec_apibank_dir: Path,
    toolalpaca_repo: str,
    bfcl_paths: list[Path],
) -> set[str]:
    """Collect all API names and function names appearing in eval sets."""
    names = set()

    # API-Bank eval (ToolSpec processed level-1/2/3)
    if toolspec_apibank_dir.exists():
        for level in ["1", "2", "3"]:
            path = toolspec_apibank_dir / f"level-{level}-api_processed.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for row in data:
                names.update(extract_names_from_system_prompt(row.get("system", "")))

    # ToolAlpaca eval (only the official test split, not the full processed data)
    try:
        from datasets import load_dataset

        ds = load_dataset(toolalpaca_repo)
        if "test" in ds:
            for row in ds["test"]:
                source_name = row.get("name")
                if source_name:
                    names.add(str(source_name).strip())
                tools = parse_toolalpaca_documentation(row.get("nl_documentation", ""))
                for t in tools:
                    names.add(t["name"])
    except Exception as e:
        print(f"Warning: could not load ToolAlpaca test split: {e}")

    # BFCL Java/JS eval sets
    for path in bfcl_paths:
        names.update(extract_names_from_bfcl(path))

    print(f"Eval contamination name set size: {len(names)}")
    return names


def is_contaminated(req: dict, eval_names: set[str]) -> bool:
    """Check if any API/function name in request appears in eval sets."""
    api_name = req.get("api_name")
    if api_name and api_name in eval_names:
        return True
    for fn in req.get("function_names", []):
        if fn in eval_names:
            return True
    return False


# ---------------------------------------------------------------------------
# Splitting and saving
# ---------------------------------------------------------------------------


def prompt_hash(req: dict) -> str:
    """Deterministic hash of the prompt (system + user)."""
    text = req["system"] + "\n" + req["user"]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_requests(
    requests: list[dict],
    train_target: int,
    dev_target: int,
    seed: int = 31082027,
) -> tuple[list[dict], list[dict]]:
    """
    Deterministic split by prompt hash.
    We assign each request a float in [0, 1) derived from its hash + seed,
    then sort and take top train_target for train, next dev_target for dev.
    """
    rng = hashlib.sha256(str(seed).encode())
    seeded = rng.hexdigest()

    scored = []
    for req in requests:
        h = prompt_hash(req)
        combined = hashlib.sha256((h + seeded).encode("utf-8")).hexdigest()
        score = int(combined, 16) / (2**256 - 1)
        scored.append((score, req))

    scored.sort(key=lambda x: x[0])
    train = [r for _, r in scored[:train_target]]
    dev = [r for _, r in scored[train_target : train_target + dev_target]]
    return train, dev


def save_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows to {path}")


def compute_overlap_stats(train: list[dict], dev: list[dict]) -> dict:
    train_ids = {r["request_id"] for r in train}
    dev_ids = {r["request_id"] for r in dev}
    train_hashes = {prompt_hash(r) for r in train}
    dev_hashes = {prompt_hash(r) for r in dev}
    return {
        "request_id_overlap": len(train_ids & dev_ids),
        "prompt_hash_overlap": len(train_hashes & dev_hashes),
        "train_count": len(train),
        "dev_count": len(dev),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    paths = get_paths()
    parser.add_argument(
        "--apibank-train",
        nargs="+",
        default=[
            str(paths["apibank_train"]),
            str(paths["apibank_train_lv2"]),
            str(paths["apibank_train_lv3"]),
        ],
        help="Path(s) to API-Bank train JSON files",
    )
    parser.add_argument(
        "--toolalpaca-repo",
        default=paths["toolalpaca_hf"],
        help="HuggingFace repo ID for ToolAlpaca",
    )
    parser.add_argument(
        "--toolspec-apibank-dir",
        default=str(paths["toolspec_apibank_dir"]),
    )
    parser.add_argument(
        "--bfcl-v3-java",
        default=str(paths["bfcl_v3_java"]),
    )
    parser.add_argument(
        "--bfcl-v3-javascript",
        default=str(paths["bfcl_v3_javascript"]),
    )
    parser.add_argument(
        "--bfcl-v4-java",
        default=str(paths["bfcl_v4_java"]),
    )
    parser.add_argument(
        "--bfcl-v4-javascript",
        default=str(paths["bfcl_v4_javascript"]),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "train" / "src_requests"))
    parser.add_argument("--apibank-train-target", type=int, default=6200)
    parser.add_argument("--apibank-dev-target", type=int, default=335)
    parser.add_argument("--toolalpaca-train-target", type=int, default=3462)
    parser.add_argument("--toolalpaca-dev-target", type=int, default=194)
    parser.add_argument("--seed", type=int, default=31082027)
    args = parser.parse_args()

    # Load
    print("Loading datasets...")
    apibank_requests = load_apibank_train([Path(p) for p in args.apibank_train])
    toolalpaca_requests = load_toolalpaca_train(args.toolalpaca_repo)

    # Build eval contamination set
    print("Building eval contamination set...")
    bfcl_paths = [
        Path(args.bfcl_v3_java),
        Path(args.bfcl_v3_javascript),
        Path(args.bfcl_v4_java),
        Path(args.bfcl_v4_javascript),
    ]
    eval_names = build_eval_name_set(
        Path(args.toolspec_apibank_dir), args.toolalpaca_repo, bfcl_paths
    )

    # Decontaminate
    print("Decontaminating...")
    apibank_clean = [r for r in apibank_requests if not is_contaminated(r, eval_names)]
    toolalpaca_clean = [r for r in toolalpaca_requests if not is_contaminated(r, eval_names)]
    print(
        f"After decontamination: API-Bank {len(apibank_clean)}, "
        f"ToolAlpaca {len(toolalpaca_clean)}"
    )

    # Split each source
    apibank_train, apibank_dev = split_requests(
        apibank_clean, args.apibank_train_target, args.apibank_dev_target, args.seed
    )
    toolalpaca_train, toolalpaca_dev = split_requests(
        toolalpaca_clean, args.toolalpaca_train_target, args.toolalpaca_dev_target, args.seed
    )

    # Combine and sort by request_id for determinism
    train = sorted(apibank_train + toolalpaca_train, key=lambda r: r["request_id"])
    dev = sorted(apibank_dev + toolalpaca_dev, key=lambda r: r["request_id"])

    # Clean internal keys before saving
    def clean(r):
        return {k: v for k, v in r.items() if k not in ("api_name", "function_names")}

    train = [clean(r) for r in train]
    dev = [clean(r) for r in dev]

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_dir / "train.jsonl", train)
    save_jsonl(output_dir / "dev.jsonl", dev)

    # Stats and manifest
    stats = {
        "apibank": {
            "train": len(apibank_train),
            "dev": len(apibank_dev),
        },
        "toolalpaca": {
            "train": len(toolalpaca_train),
            "dev": len(toolalpaca_dev),
        },
        "total": {
            "train": len(train),
            "dev": len(dev),
        },
        "overlap": compute_overlap_stats(train, dev),
    }
    print(json.dumps(stats, indent=2))

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "created": str(Path.cwd()),
        "args": vars(args),
        "stats": stats,
        "eval_name_set_size": len(eval_names),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
