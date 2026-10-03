#!/usr/bin/env python3
"""Download external datasets used by OoO-Spec.

All files are placed under this repository's data directory:

  data/raw/                 original training datasets
  data/evalsets/            evaluation datasets consumed by converters

Some upstream projects change file layouts over time. The script therefore
downloads complete upstream snapshots when necessary, then copies known files
into the canonical layout if they are present.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent
RAW_DIR = DATA_DIR / "raw"
EVALSETS_DIR = DATA_DIR / "evalsets"
DOWNLOADS_DIR = DATA_DIR / "downloads"

HF_TOOLALPACA_REPO = "Ahren09/ToolAlpaca"
HF_BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
HF_OPENFUNCTIONS_REPO = "Post-training-Data-Flywheel/gorilla-openfunctions-v1"

APIBANK_GIT_REPO = "https://github.com/AlibabaResearch/DAMO-ConvAI.git"
SEALTOOLS_GIT_REPO = "https://github.com/fairyshine/Seal-Tools.git"


def log(message: str) -> None:
    print(f"[download] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_hf_hub():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with "
            "`pip install huggingface_hub` or use --skip-hf datasets."
        ) from exc
    return snapshot_download


def download_url(url: str, destination: Path, overwrite: bool = False) -> None:
    ensure_dir(destination.parent)
    if destination.exists() and not overwrite:
        log(f"exists: {destination}")
        return
    log(f"url: {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def clone_or_pull(repo_url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    if (destination / ".git").exists():
        log(f"updating git repo: {destination}")
        subprocess.run(["git", "-C", str(destination), "pull", "--ff-only"], check=True)
        return
    if destination.exists():
        log(f"using existing directory: {destination}")
        return
    log(f"cloning: {repo_url} -> {destination}")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(destination)], check=True)


def hf_snapshot(repo_id: str, destination: Path, repo_type: str = "dataset") -> Path:
    snapshot_download = require_hf_hub()
    ensure_dir(destination)
    log(f"huggingface snapshot: {repo_id} -> {destination}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )
    return destination


def copy_first_existing(root: Path, candidates: list[str], destination: Path) -> bool:
    for candidate in candidates:
        matches = list(root.rglob(candidate))
        if matches:
            ensure_dir(destination.parent)
            shutil.copy2(matches[0], destination)
            log(f"copied: {matches[0]} -> {destination}")
            return True
    log(f"missing: {destination.name} under {root}")
    return False


def download_apibank(args: argparse.Namespace) -> None:
    if args.apibank_train_url:
        download_url(args.apibank_train_url, RAW_DIR / "api_bank" / "lv1-train.json", args.overwrite)

    repo_dir = DOWNLOADS_DIR / "DAMO-ConvAI"
    clone_or_pull(APIBANK_GIT_REPO, repo_dir)

    targets = {
        "lv1-train.json": RAW_DIR / "api_bank" / "lv1-train.json",
        "lv2-train.json": RAW_DIR / "api_bank" / "lv2-train.json",
        "lv3-train.json": RAW_DIR / "api_bank" / "lv3-train.json",
    }
    for filename, destination in targets.items():
        if destination.exists() and not args.overwrite:
            continue
        copy_first_existing(repo_dir, [filename], destination)


def download_toolalpaca(args: argparse.Namespace) -> None:
    if args.skip_hf:
        return
    hf_snapshot(HF_TOOLALPACA_REPO, DOWNLOADS_DIR / "toolalpaca_hf")


def download_bfcl(args: argparse.Namespace) -> None:
    if args.skip_hf:
        return
    snapshot = hf_snapshot(HF_BFCL_REPO, DOWNLOADS_DIR / "bfcl_hf")
    mappings = {
        "BFCL_v3_java.json": EVALSETS_DIR / "bfcl" / "BFCL_v3_java.json",
        "BFCL_v3_javascript.json": EVALSETS_DIR / "bfcl" / "BFCL_v3_javascript.json",
        "BFCL_v4_simple_java.json": EVALSETS_DIR / "bfcl_v4" / "BFCL_v4_simple_java.json",
        "BFCL_v4_simple_javascript.json": EVALSETS_DIR / "bfcl_v4" / "BFCL_v4_simple_javascript.json",
        "BFCL_v4_simple_python.json": EVALSETS_DIR / "bfcl_v4" / "BFCL_v4_simple_python.json",
    }
    for filename, destination in mappings.items():
        if destination.exists() and not args.overwrite:
            continue
        copy_first_existing(snapshot, [filename], destination)


def download_openfunctions(args: argparse.Namespace) -> None:
    if args.skip_hf:
        return
    snapshot = hf_snapshot(HF_OPENFUNCTIONS_REPO, DOWNLOADS_DIR / "openfunctions_hf")
    destination = EVALSETS_DIR / "openfunctions" / "gorilla_openfunctions_v1_test.json"
    if destination.exists() and not args.overwrite:
        return
    copy_first_existing(snapshot, ["gorilla_openfunctions_v1_test.json", "*.json"], destination)


def download_sealtools(args: argparse.Namespace) -> None:
    repo_dir = DOWNLOADS_DIR / "Seal-Tools"
    clone_or_pull(SEALTOOLS_GIT_REPO, repo_dir)
    mappings = {
        "test_in_domain.json": EVALSETS_DIR / "sealtools" / "test_in_domain.json",
        "test_out_domain.json": EVALSETS_DIR / "sealtools" / "test_out_domain.json",
    }
    for filename, destination in mappings.items():
        if destination.exists() and not args.overwrite:
            continue
        copy_first_existing(repo_dir, [filename], destination)


def download_optional_urls(args: argparse.Namespace) -> None:
    optional = [
        (args.mobile_actions_url, EVALSETS_DIR / "mobile_actions_dataset.jsonl"),
        (args.bfcl_v3_simple_url, EVALSETS_DIR / "bfcl_v3_train.parquet"),
    ]
    for url, destination in optional:
        if url:
            download_url(url, destination, args.overwrite)


def write_manifest() -> None:
    manifest = {
        "raw_dir": str(RAW_DIR.relative_to(PROJECT_ROOT)),
        "evalsets_dir": str(EVALSETS_DIR.relative_to(PROJECT_ROOT)),
        "downloads_dir": str(DOWNLOADS_DIR.relative_to(PROJECT_ROOT)),
        "sources": {
            "apibank": APIBANK_GIT_REPO,
            "toolalpaca": f"https://huggingface.co/datasets/{HF_TOOLALPACA_REPO}",
            "bfcl": f"https://huggingface.co/datasets/{HF_BFCL_REPO}",
            "openfunctions": f"https://huggingface.co/datasets/{HF_OPENFUNCTIONS_REPO}",
            "sealtools": SEALTOOLS_GIT_REPO,
        },
    }
    manifest_path = DATA_DIR / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"wrote manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing canonical files.")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace-hosted datasets.")
    parser.add_argument("--apibank-train-url", default=os.environ.get("OOOSPEC_APIBANK_TRAIN_URL"))
    parser.add_argument("--mobile-actions-url", default=os.environ.get("OOOSPEC_MOBILE_ACTIONS_URL"))
    parser.add_argument("--bfcl-v3-simple-url", default=os.environ.get("OOOSPEC_BFCL_V3_SIMPLE_URL"))
    return parser.parse_args()


def main() -> None:
    ensure_dir(RAW_DIR)
    ensure_dir(EVALSETS_DIR)
    ensure_dir(DOWNLOADS_DIR)

    args = parse_args()
    download_apibank(args)
    download_toolalpaca(args)
    download_bfcl(args)
    download_openfunctions(args)
    download_sealtools(args)
    download_optional_urls(args)
    write_manifest()

    log("done")


if __name__ == "__main__":
    main()
