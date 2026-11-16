#!/usr/bin/env bash
# Container entrypoint: subcommands go to the pipeline, anything else runs as-is.
set -euo pipefail

MODEL_DIR="${HF_HOME:-/models/hf}/hub/models--Systran--faster-whisper-large-v3"

case "${1:-}" in
  run|gate|route|repair|summarize|verify|status)
    if [[ "${1}" == "run" && ! -d "$MODEL_DIR" ]]; then
      printf 'Нет модели whisper large-v3 в %s\n' "$MODEL_DIR" >&2
      printf 'Подготовьте её на хосте: bash docker/prepare-models.sh (см. DOCKER.md)\n' >&2
      exit 64
    fi
    exec /app/.venv/bin/python /app/asr_pipeline.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
