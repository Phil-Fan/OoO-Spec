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
import sys
import threading
import time
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
)
from evaluation.inference_toolspec import toolspec_forward  # noqa: E402
from model.toolspec.modeling_qwen_kv import Qwen2ForCausalLM  # noqa: E402
from model.toolspec.schema_fsm import SchemaFSM  # noqa: E402


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
):
    sidecar_time = 0.0
    raw_hint = ""
    formatted_hint = None
    hint_tokens = None

    if use_hint:
        if prefetched_hint is not None:
            raw_hint, sidecar_time = prefetched_hint
        else:
            raw_hint, sidecar_time = generate_sidecar_hint(sidecar_tok, src_row)
        formatted_hint = parse_sidecar_output(raw_hint)
        if formatted_hint is not None:
            hint_tokens = target_tok.encode(formatted_hint, add_special_tokens=False)
            hint_tokens = torch.tensor(hint_tokens, dtype=torch.long)

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
    output_ids, new_token, step, accept_lengths, _ = toolspec_forward(
        inputs,
        [],
        schema_fsm,
        target_model,
        target_tok,
        max_new_tokens=max_new_tokens,
        output_id_topk=8,
        adj_matrix=adj_matrix,
        sidecar_hint=hint_tokens,
    )
    torch.cuda.synchronize(target_model.device)
    wall_time = time.time() - start
    output_text = decode_target_output(target_tok, output_ids, len(inputs["input_ids"][0]))

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
