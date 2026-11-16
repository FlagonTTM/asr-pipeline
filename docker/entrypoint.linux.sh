#!/usr/bin/env bash
# Linux ensemble entrypoint: fail early if the model volume is empty.
set -euo pipefail

MODELS="${HF_HOME:-/models/hf}/hub"
case "${1:-}" in
  run|gate|route|repair|summarize|verify|status)
    if [[ ! -d "$MODELS" ]] || [[ -z "$(ls -A "$MODELS" 2>/dev/null)" ]]; then
      printf 'Том моделей пуст: %s\n' "$MODELS" >&2
      printf 'Скачайте веса на хосте: HF_HOME=./models/linux/hf python -m ... (см. DOCKER.md)\n' >&2
      exit 64
    fi
    exec /app/.venv/bin/python /app/asr_pipeline.py "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
