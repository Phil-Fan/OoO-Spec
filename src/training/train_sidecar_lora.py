#!/usr/bin/env python3
"""
Train OoO-Spec sidecar with LoRA on Qwen3-0.6B.

Config from plan/repro.md §3.1:
  - Base: Qwen/Qwen3-0.6B
  - LoRA: r=32, alpha=64, dropout=0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Epochs: 1
  - Effective batch: 32 (micro 4 x accum 8)
  - LR: 1e-4, cosine, 5% warmup
  - Max seq len: 2048
  - Seed: 31082027
  - Save at step 2362
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def get_paths():
    root = Path(__file__).resolve().parents[2]
    return {
        "model_name": os.environ.get("OOOSPEC_SIDECAR_BASE", "Qwen/Qwen3-0.6B"),
        "train_rows": root / "data" / "train" / "rows" / "train.jsonl",
        "dev_rows": root / "data" / "train" / "rows" / "dev.jsonl",
        "output_dir": root / "outputs" / "sidecar-lora",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=get_paths()["model_name"])
    parser.add_argument("--train-rows", default=str(get_paths()["train_rows"]))
    parser.add_argument("--dev-rows", default=str(get_paths()["dev_rows"]))
    parser.add_argument("--output-dir", default=str(get_paths()["output_dir"]))

    # LoRA
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31082027)
    parser.add_argument("--save-steps", type=int, default=2362)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--disable-cache", action="store_true", default=True)

    return parser.parse_args()


def load_rows(path: Path):
    rows = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    return Dataset.from_list(rows)


def prepare_example(example, tokenizer, max_seq_len: int):
    """Tokenize and create labels with prompt tokens masked."""
    messages = example["messages"]
    # Use Qwen3 no-thinking template
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
    )

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    full_tokens = tokenizer(full_text, truncation=True, max_length=max_seq_len, add_special_tokens=False)

    input_ids = full_tokens["input_ids"]
    labels = [-100] * len(input_ids)

    # Mask prompt tokens: labels start after prompt tokens
    prompt_len = len(prompt_tokens["input_ids"])
    if prompt_len < len(input_ids):
        labels[prompt_len:] = input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Load tokenizer and model
    print(f"Loading model {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        use_cache=not args.disable_cache,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print("Loading semantic rows...")
    train_ds = load_rows(Path(args.train_rows))
    dev_ds = load_rows(Path(args.dev_rows))

    print(f"Train rows: {len(train_ds)}, Dev rows: {len(dev_ds)}")

    # Tokenize
    def tokenize_fn(ex):
        return prepare_example(ex, tokenizer, args.max_seq_len)

    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(tokenize_fn, remove_columns=dev_ds.column_names)

    # Training args
    total_steps = (len(train_ds) + args.per_device_batch * args.grad_accum - 1) // (
        args.per_device_batch * args.grad_accum
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Estimated total steps: {total_steps}, warmup_steps: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(Path(args.output_dir) / "final")
    print(f"Training complete. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
