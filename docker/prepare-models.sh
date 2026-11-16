#!/usr/bin/env bash
# Prepare the Whisper large-v3 CT2 model for the container.
# Copies from the host HF cache when present; otherwise downloads once (needs network).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODELS="${MODELS:-$ROOT/models}"
SRC="${HF_CACHE:-$HOME/.cache/huggingface/hub}"
NAME="models--Systran--faster-whisper-large-v3"
DEST="$MODELS/hf/hub"

mkdir -p "$DEST"
if [[ -d "$SRC/$NAME" ]]; then
  printf 'Копирую %s -> %s (может занять пару минут)\n' "$SRC/$NAME" "$DEST/"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$SRC/$NAME" "$DEST/"
  else
    cp -R "$SRC/$NAME" "$DEST/"
  fi
else
  printf 'Локального кеша нет — качаю модель в контейнере (нужна сеть)\n'
  docker run --rm \
    -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
    -v "$MODELS:/models" \
    --entrypoint /app/.venv/bin/python \
    asr-pipeline:0.1 - <<'PY'
from faster_whisper import WhisperModel
WhisperModel("large-v3", device="cpu", compute_type="int8", local_files_only=False)
print("model downloaded")
PY
fi

printf 'Проверяю, что модель читается офлайн\n'
docker run --rm -v "$MODELS:/models" \
  --entrypoint /app/.venv/bin/python asr-pipeline:0.1 - <<'PY'
from faster_whisper import WhisperModel
m = WhisperModel("large-v3", device="cpu", compute_type="int8", local_files_only=True)
print("model ok")
PY
