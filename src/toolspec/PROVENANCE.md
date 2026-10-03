# Provenance

This directory contains **third-party code vendored from ToolSpec**. It is
included so that the OoO-Spec reproduction can import ToolSpec's schema FSM,
candidate-tree construction, retrieval, and KV-cache decoding modules. It is
**not** original work of this repository.

| Field | Value |
|---|---|
| Upstream project | [hemingkx/ToolSpec](https://github.com/hemingkx/ToolSpec) |
| Pinned revision | `69335b9` (see `plan/repro.md`) |
| License | Apache License 2.0 (see `LICENSE` in this directory) |
| Copyright | Heming Xia, Yongqi Li, Cunxiao Du, Mingbo Song, Wenjie Li |
| Paper | ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding, arXiv:2604.13519 |

If you reuse this code, follow the upstream project and cite the ToolSpec paper.

Notes:

- The vendored tree may contain the upstream authors' example/default paths
  (for example under `model/samd` and `evaluation/speed.py`). These are left as
  found for fidelity with upstream; override them with CLI flags or environment
  variables when running.
- Integration glue for OoO-Spec lives outside this directory, in
  `src/inference/` and `src/preprocessing/`. The OoO-Spec sidecar logic is not
  part of the vendored upstream code.
