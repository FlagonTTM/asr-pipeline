#!/usr/bin/env bash
# Download Linux-ensemble weights into ./models/linux/hf.
# HF is slow in some networks; HF_ENDPOINT=https://hf-mirror.com is ~4x faster here.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export HF_HOME="${HF_HOME:-$ROOT/models/linux/hf}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME"

PY="${PY:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  printf 'Нет %s. Укажите PY=/path/to/python с huggingface_hub.\n' "$PY" >&2
  exit 64
fi

"$PY" - <<'PY'
import os, time
from huggingface_hub import snapshot_download

repos = [
    ("ai-sage/GigaAM-v3", ["*.gguf", "*.nemo"]),
    ("Systran/faster-whisper-large-v3", []),
    ("nvidia/parakeet-tdt-0.6b-v3", ["*.gguf", "*.nemo", "plots/*", ".eval_results/*"]),
    ("Qwen/Qwen3-ASR-1.7B", []),
]
only = os.environ.get("ONLY", "").split(",") if os.environ.get("ONLY") else None
for repo, ignore in repos:
    if only and repo not in only:
        continue
    t = time.time()
    print(f"=== {repo}", flush=True)
    path = snapshot_download(repo, max_workers=4, ignore_patterns=ignore)
    print(f"OK {time.time() - t:.0f}s {path}", flush=True)
PY

du -sh "$HF_HOME"
