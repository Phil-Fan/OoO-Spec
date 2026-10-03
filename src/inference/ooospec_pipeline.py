#!/usr/bin/env python3
"""
End-to-end OoO-Spec test: sidecar (OoO-Spec) + ToolSpec target.

Loads the LoRA sidecar on one GPU and the ToolSpec target on another,
generates a sidecar hint for each sample, then runs ToolSpec with the
hint injected into its retrieval tree.
"""

import json
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
# Make src/toolspec importable for ToolSpec modules.
TOOLSPECT_ROOT = SRC_ROOT / "toolspec"
sys.path.insert(0, str(TOOLSPECT_ROOT))

from preprocessing.expand_semantic_rows import (  # noqa: E402
    SYSTEM_INSTRUCTION as SIDECAR_SYSTEM,
    format_schemas_with_indices,
    format_schemas_with_param_tags,
)
from evaluation.inference_toolspec import toolspec_forward  # noqa: E402
from model.toolspec.modeling_qwen_kv import Qwen2ForCausalLM  # noqa: E402
from model.toolspec.schema_fsm import SchemaFSM  # noqa: E402

# Persistent HTTP session: reuse the connection to the sidecar service.
_HTTP_SESSION = requests.Session()


def get_paths():
    return {
        "sidecar_base": os.environ.get("OOOSPEC_SIDECAR_BASE", "Qwen/Qwen3-0.6B"),
        "sidecar_adapter": ROOT / "outputs" / "sidecar-lora" / "final",
        "target_model": os.environ.get("OOOSPEC_TARGET_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "src_dev": ROOT / "data" / "train" / "src_requests" / "dev.jsonl",
    }


def load_src_requests(path: Path):
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]


def build_sidecar_direct_call_prompt(src_row: dict) -> list[dict]:
    """Build the same direct-call conversation used during LoRA training."""
    schemas = src_row.get("schemas", [])
    dialogue = src_row.get("user", "").replace(
        "\n\n<user> Based on our conversation above, please only make one tool call to solve my need.</user>",
        "",
    )
    user_prompt = (
        f"{dialogue}\n\n"
        "Available tools:\n"
        f"{format_schemas_with_indices(schemas)}\n\n"
        "What is the complete tool call? Respond with a JSON object."
    )
    return [
        {"role": "system", "content": SIDECAR_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


def build_sidecar_prompt_text(sidecar_tok, src_row: dict) -> str:
    """Return the templated prompt string for the sidecar."""
    messages = build_sidecar_direct_call_prompt(src_row)
    return sidecar_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_sidecar_hint(sidecar_tok, src_row: dict, timeout: float = 30.0):
    prompt = build_sidecar_prompt_text(sidecar_tok, src_row)
    start = time.time()
    resp = requests.post(
        SIDECAR_API_URL,
        json={
            "model": "sidecar",
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0,
        },
        timeout=timeout,
    )
    elapsed = time.time() - start
    resp.raise_for_status()
    text = resp.json()["choices"][0]["text"].strip()
    return text, elapsed


class SidecarPrefetcher:
    """Background prefetcher for sidecar hints.

    Submit the prompt for sample i+1 while sample i is being processed by the
    target model, so that the sidecar network/GPU time overlaps with target
    generation.
    """

    def __init__(self, api_url: str, max_queue: int = 2):
        self.api_url = api_url
        self._pending = queue.Queue(maxsize=max_queue)
        self._results = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while not self._stop.is_set():
            try:
                request_id, prompt = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue
            start = time.time()
            try:
                resp = requests.post(
                    self.api_url,
                    json={
                        "model": "sidecar",
                        "prompt": prompt,
                        "max_tokens": 256,
                        "temperature": 0,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["text"].strip()
                elapsed = time.time() - start
                self._results.put((request_id, text, elapsed, None))
            except Exception as exc:
                elapsed = time.time() - start
                self._results.put((request_id, "", elapsed, exc))

    def submit(self, request_id: str, prompt: str):
        """Enqueue a sidecar request. Blocks if the prefetch queue is full."""
        self._pending.put((request_id, prompt), block=True)

    def get(self, request_id: str):
        """Block until the requested sidecar result is available.

        Returns (raw_hint_text, elapsed_seconds). Re-raises any worker exception.
        """
        while True:
            rid, text, elapsed, exc = self._results.get(block=True)
            if rid == request_id:
                if exc is not None:
                    raise exc
                return text, elapsed
            # Should not happen if used sequentially, but drain just in case.

    def close(self):
        self._stop.set()
        self._worker.join(timeout=5)


def parse_sidecar_output(text: str):
    """Parse compact JSON and reformat to the target-friendly spaced JSON."""
    text = text.strip()
    if not text:
        return None
    # If the model added extra chatter, grab the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    params = obj.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
    return json.dumps(
        {"name": name, "parameters": params},
        ensure_ascii=False,
        separators=(", ", ": "),
    )


SIDECAR_API_URL = "http://localhost:7892/v1/completions"

# Trailing instruction present in the source requests; stripped before building
# the sidecar prompts (matches the offline training rows).
SIDECAR_TAIL = (
    "\n\n<user> Based on our conversation above, please only make one tool call "
    "to solve my need.</user>"
)


def _sidecar_chat(sidecar_tok, user_text: str) -> str:
    return sidecar_tok.apply_chat_template(
        [
            {"role": "system", "content": SIDECAR_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_sidecar_slot_prompts(sidecar_tok, src_row: dict):
    """Paper-style online queries: one function-index prompt plus one prompt per
    (candidate function, parameter) slot.

    Returns ``(fn_prompt, [(f_idx, p_name, prompt), ...])``.
    """
    schemas = src_row.get("schemas", [])
    dialogue = src_row.get("user", "").replace(SIDECAR_TAIL, "")
    fn_user = (
        f"{dialogue}\n\nAvailable tools:\n"
        f"{format_schemas_with_indices(schemas)}\n\n"
        "Which tool should be called? Respond with only the index number."
    )
    fn_prompt = _sidecar_chat(sidecar_tok, fn_user)

    arg_prompts = []
    param_tags = format_schemas_with_param_tags(schemas)
    for f_idx, s in enumerate(schemas):
        for p_name in (s.get("parameters") or {}):
            user = (
                f"{dialogue}\n\nAvailable tool parameters:\n{param_tags}\n\n"
                f"What is the value for parameter '{p_name}' of tool [{f_idx}] {s['name']}? "
                "Respond with a compact JSON value, or the literal string null if not applicable."
            )
            arg_prompts.append((f_idx, p_name, _sidecar_chat(sidecar_tok, user)))
    return fn_prompt, arg_prompts


def _parse_int_index(text: str, n_tools: int):
    match = re.search(r"-?\d+", text or "")
    if not match:
        return None
    index = int(match.group())
    return index if 0 <= index < n_tools else None


def _parse_json_value(text: str):
    """Return ``(value, ok)`` for a compact JSON value or the literal ``null``."""
    text = (text or "").strip()
    if not text:
        return None, False
    if text.lower().startswith("null"):
        return None, True
    start = None
    for ch in ('"', "{", "["):
        pos = text.find(ch)
        if pos != -1 and (start is None or pos < start):
            start = pos
    if start is None:
        match = re.match(r"-?\d+(?:\.\d+)?|true|false", text)
        if not match:
            return None, False
        try:
            return json.loads(match.group()), True
        except json.JSONDecodeError:
            return None, False
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value, True
    except json.JSONDecodeError:
        return None, False


def _join_normalized_call(schemas, fn_idx, arg_values):
    if not isinstance(fn_idx, int) or not (0 <= fn_idx < len(schemas)):
        return None
    params = {k: v for k, v in (arg_values or {}).items() if v is not None}
    return {"name": schemas[fn_idx]["name"], "parameters": params}


def _render_hint_text(call):
    if not call:
        return None
    return json.dumps(call, ensure_ascii=False, separators=(", ", ": "))


def _render_call_views(call):
    """Render one normalized call into equivalent textual views.

    The first view is the target-native spaced JSON (used for direct candidate
    injection); the extra views broaden the hint bank for suffix matching.
    """
    if not call:
        return []
    plain = json.dumps(call, ensure_ascii=False, separators=(", ", ": "))
    markdown = f"```json\n{plain}\n```"
    xml = f"<tool_call>\n{plain}\n</tool_call>"
    return [plain, markdown, xml]


class SidecarHintFuture:
    """Within-request asynchronous sidecar job (paper-style parallel slot wave).

    The job is launched at request arrival and the target polls it
    non-blockingly at candidate-construction boundaries (``poll()``). The
    sidecar wall time is exposed via ``elapsed``; the target never waits.
    """

    def __init__(self, sidecar_tok, src_row, api_url=SIDECAR_API_URL, max_workers=16):
        self._ready = threading.Event()
        self._text = None
        self._views = []
        self._elapsed = 0.0
        self._exc = None
        self._consumed = False
        self._thread = threading.Thread(
            target=self._run,
            args=(sidecar_tok, src_row, api_url, max_workers),
            daemon=True,
        )
        self._thread.start()

    def _post(self, prompt: str) -> str:
        resp = _HTTP_SESSION.post(
            self.api_url,
            json={"model": "sidecar", "prompt": prompt, "max_tokens": 32, "temperature": 0},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    def _run(self, sidecar_tok, src_row, api_url, max_workers):
        self.api_url = api_url
        try:
            start = time.time()
            schemas = src_row.get("schemas", [])
            fn_prompt, arg_prompts = build_sidecar_slot_prompts(sidecar_tok, src_row)
            prompts = [fn_prompt] + [prompt for _, _, prompt in arg_prompts]
            texts = [""] * len(prompts)
            with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as executor:
                futures = {executor.submit(self._post, p): i for i, p in enumerate(prompts)}
                for future in as_completed(futures):
                    texts[futures[future]] = future.result()

            fn_idx = _parse_int_index(texts[0], len(schemas))
            arg_values = {}
            if fn_idx is not None:
                for (f_idx, p_name, _), text in zip(arg_prompts, texts[1:]):
                    if f_idx != fn_idx:
                        continue
                    value, ok = _parse_json_value(text)
                    if ok and value is not None:
                        arg_values[p_name] = value
            call = _join_normalized_call(schemas, fn_idx, arg_values)
            self._views = _render_call_views(call)
            self._text = self._views[0] if self._views else None
            self._elapsed = time.time() - start
        except Exception as exc:  # noqa: BLE001
            self._exc = exc
        finally:
            self._ready.set()

    def poll(self):
        """Non-blocking pickup; returns ``(views, elapsed)`` once, else ``None``."""
        if not self._ready.is_set() or self._consumed:
            return None
        self._consumed = True
        if self._exc is not None:
            return None
        return (self._views, self._elapsed)

    @property
    def elapsed(self) -> float:
        return self._elapsed

    def wait(self, timeout=None):
        self._ready.wait(timeout)
        if self._exc is not None:
            raise self._exc
        return (self._views, self._elapsed)


def load_sidecar_tokenizer(base_path: str):
    print(f"[Sidecar] loading tokenizer {base_path}")
    tok = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_target(model_path: str, device: str):
    print(f"[Target/ToolSpec] loading {model_path} on {device}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Backward-compat for custom ToolSpec model code written against older transformers.
    if not hasattr(config, "rope_theta"):
        config.rope_theta = 1000000.0
    if config.model_type == "qwen2":
        model_cls = Qwen2ForCausalLM
    else:
        raise ValueError(f"Unsupported target model type: {config.model_type}")
    model = model_cls.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device,
    )
    # Attach a generation_config so the custom ToolSpec model code can read eos_token_id.
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except Exception:
        model.generation_config = GenerationConfig(eos_token_id=tok.eos_token_id)
    return model, tok


def load_vanilla_target(model_path: str, device: str):
    print(f"[Target/Vanilla] loading {model_path} on {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device,
    )
    return model


def decode_target_output(target_tok, output_ids, input_len):
    output_ids = output_ids[0, input_len:]
    output_text = target_tok.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    for special_token in target_tok.special_tokens_map.values():
        if isinstance(special_token, list):
            for st in special_token:
                output_text = output_text.replace(st, "")
        else:
            output_text = output_text.replace(special_token, "")
    return output_text.strip()


def run_toolspec_sample(
    src_row,
    sidecar_tok,
    target_model,
    target_tok,
    adj_matrix,
    max_new_tokens: int = 256,
    use_hint: bool = True,
    prefetched_hint: tuple | None = None,
    output_memory: list | None = None,
    sidecar_future: "SidecarHintFuture | None" = None,
    slot_mode: bool = False,
):
    sidecar_time = 0.0
    raw_hint = ""
    formatted_hint = None
    hint_tokens = None
    provider = None

    if use_hint:
        if sidecar_future is not None:
            provider = sidecar_future
        elif slot_mode:
            # Within-request async: launch the parallel slot wave now, let the
            # target poll it non-blockingly while it decodes.
            provider = SidecarHintFuture(sidecar_tok, src_row)
        elif prefetched_hint is not None:
            raw_hint, sidecar_time = prefetched_hint
            formatted_hint = parse_sidecar_output(raw_hint)
            if formatted_hint is not None:
                hint_tokens = torch.tensor(
                    target_tok.encode(formatted_hint, add_special_tokens=False), dtype=torch.long
                )
        else:
            raw_hint, sidecar_time = generate_sidecar_hint(sidecar_tok, src_row)
            formatted_hint = parse_sidecar_output(raw_hint)
            if formatted_hint is not None:
                hint_tokens = torch.tensor(
                    target_tok.encode(formatted_hint, add_special_tokens=False), dtype=torch.long
                )

    messages = src_row["messages"]
    system = src_row["system"]
    prompt_text = target_tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = target_tok([prompt_text], return_tensors="pt").to(target_model.device)
    schema_fsm = SchemaFSM(system, target_tok)

    torch.cuda.synchronize(target_model.device)
    start = time.time()
    output_ids, new_token, step, accept_lengths, question_hidden_state = toolspec_forward(
        inputs,
        output_memory if output_memory is not None else [],
        schema_fsm,
        target_model,
        target_tok,
        max_new_tokens=max_new_tokens,
        output_id_topk=8,
        adj_matrix=adj_matrix,
        sidecar_hint=hint_tokens,
        sidecar_hint_provider=provider,
    )
    torch.cuda.synchronize(target_model.device)
    wall_time = time.time() - start
    output_text = decode_target_output(target_tok, output_ids, len(inputs["input_ids"][0]))

    if provider is not None:
        # Measurement only: make sure the sidecar wall time is captured even if
        # the target finished before consuming the hint. Target wall_time was
        # already recorded above, so this does not affect it.
        try:
            provider.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        sidecar_time = provider.elapsed
    sidecar_wait = 0.0  # non-blocking pickup; the target never waits for a hint

    if output_memory is not None:
        gen_ids = output_ids[0, len(inputs["input_ids"][0]):]
        output_memory.append(
            {
                "question_id": src_row.get("request_id"),
                "question_hidden_state": question_hidden_state,
                "output_ids": gen_ids.tolist(),
            }
        )

    return {
        "request_id": src_row.get("request_id"),
        "sidecar_raw": raw_hint,
        "sidecar_formatted": formatted_hint,
        "target_output": output_text,
        "new_tokens": int(new_token),
        "decoding_steps": int(step),
        "mean_accept": float(sum(accept_lengths) / len(accept_lengths)) if accept_lengths else 0.0,
        "wall_time": wall_time,
        "sidecar_time": sidecar_time,
        "sidecar_wait": sidecar_wait,
    }


def run_vanilla_sample(src_row, vanilla_model, target_tok, max_new_tokens: int = 256):
    prompt_text = target_tok.apply_chat_template(
        src_row["messages"],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = target_tok([prompt_text], return_tensors="pt").to(vanilla_model.device)

    torch.cuda.synchronize(vanilla_model.device)
    start = time.time()
    with torch.no_grad():
        output_ids = vanilla_model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=target_tok.pad_token_id,
        )
    torch.cuda.synchronize(vanilla_model.device)
    wall_time = time.time() - start
    output_text = decode_target_output(target_tok, output_ids, len(inputs["input_ids"][0]))
    new_token = output_ids.size(1) - inputs["input_ids"].size(1)

    return {
        "target_output": output_text,
        "new_tokens": int(new_token),
        "decoding_steps": int(new_token),
        "mean_accept": 1.0,
        "wall_time": wall_time,
    }


def main():
    paths = get_paths()
    src_rows = load_src_requests(paths["src_dev"])

    target_device = "cuda:0"
    vanilla_device = "cuda:2"

    sidecar_tok = load_sidecar_tokenizer(paths["sidecar_base"])
    target_model, target_tok = load_target(paths["target_model"], target_device)
    vanilla_model = load_vanilla_target(paths["target_model"], vanilla_device)

    adj_matrix_hint = torch.zeros(
        (target_model.vocab_size, 8),
        dtype=torch.long,
        device=target_model.device,
        requires_grad=False,
    )
    adj_matrix_baseline = torch.zeros_like(adj_matrix_hint)

    num_samples = min(3, len(src_rows))
    for i in range(num_samples):
        print(f"\n========== Sample {i} ==========")
        vanilla_res = run_vanilla_sample(src_rows[i], vanilla_model, target_tok)
        toolspec_res = run_toolspec_sample(
            src_rows[i],
            sidecar_tok,
            target_model,
            target_tok,
            adj_matrix_baseline,
            use_hint=False,
        )
        ooospec_res = run_toolspec_sample(
            src_rows[i],
            sidecar_tok,
            target_model,
            target_tok,
            adj_matrix_hint,
            use_hint=True,
        )

        t_v = vanilla_res["wall_time"]
        t_t = toolspec_res["wall_time"]
        t_ooo = ooospec_res["wall_time"]
        t_ooo_total = t_ooo + ooospec_res["sidecar_time"]

        print(f"request_id: {ooospec_res['request_id']}")
        print(f"sidecar raw:       {ooospec_res['sidecar_raw']}")
        print(f"sidecar formatted: {ooospec_res['sidecar_formatted']}")
        print(f"Vanilla   -> {vanilla_res['target_output']}")
        print(f"  tokens={vanilla_res['new_tokens']}, steps={vanilla_res['decoding_steps']}, time={t_v:.3f}s")
        print(f"ToolSpec  -> {toolspec_res['target_output']}")
        print(f"  tokens={toolspec_res['new_tokens']}, steps={toolspec_res['decoding_steps']}, mean_accept={toolspec_res['mean_accept']:.2f}, time={t_t:.3f}s")
        print(f"OoO-Spec  -> {ooospec_res['target_output']}")
        print(f"  tokens={ooospec_res['new_tokens']}, steps={ooospec_res['decoding_steps']}, mean_accept={ooospec_res['mean_accept']:.2f}, time={t_ooo:.3f}s (sidecar {ooospec_res['sidecar_time']:.3f}s, total {t_ooo_total:.3f}s)")
        print(f"Speedups  ToolSpec/vanilla={t_v/t_t:.2f}x,  OoO(target)/vanilla={t_v/t_ooo:.2f}x,  OoO(total)/vanilla={t_v/t_ooo_total:.2f}x,  OoO/ToolSpec={t_t/t_ooo:.2f}x")


if __name__ == "__main__":
    main()
