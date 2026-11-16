# Запуск в Docker

Контейнер фиксирует Python 3.12, ffmpeg и версии библиотек, поэтому на другой машине
поведение то же, что здесь. Модель монтируется томом, а не запекается в образ:
её хеши и происхождение записаны в `MODEL_MANIFEST.md`.

## Что внутри

- `python:3.12-slim` + `ffmpeg`
- `/app/.venv` с зафиксированными пакетами из `requirements-docker.txt`
- код пайплайна и `engines/whisper_ct2.py`
- офлайн по умолчанию: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`

## Сборка

```bash
docker build -t asr-pipeline:0.1 .
```

## Модель

```bash
bash docker/prepare-models.sh          # копирует из ~/.cache/huggingface
# или, если кеша нет и есть сеть:
docker run --rm -e HF_HUB_OFFLINE=0 -v "$PWD/models:/models" \
  --entrypoint /app/.venv/bin/python asr-pipeline:0.1 - <<'PY'
from faster_whisper import WhisperModel
WhisperModel("large-v3", local_files_only=False)
PY
```

Модель — `Systran/faster-whisper-large-v3` (CT2, int8 на CPU). Том `./models` монтируется в `/models`.

## Запуск

```bash
mkdir -p runs && cp /path/to/audio.wav runs/audio.wav
docker compose run --rm asr
```

Или напрямую:

```bash
docker run --rm -v "$PWD/models:/models" -v "$PWD/runs:/runs" asr-pipeline:0.1 \
  run --audio /runs/audio.wav --out /runs/out \
  --engine whisper_ct2 --engines /app/docker/engines.docker.json \
  --vad energy --lid declared
```

`--lid declared` не требует второго прохода Whisper: язык берётся из флага. Для честного LID
уберите флаг, но тогда в томе нужна ещё и модель LID (по умолчанию та же large-v3).

## Проверка без модели и без ML

```bash
docker run --rm asr-pipeline:0.1 bash scripts/smoke.sh
```

Ожидается `SMOKE PASS` и коды 2 на gate/route/summary.

## Что в контейнер НЕ входит

- **MLX-движки** (Whisper-turbo, Parakeet, Qwen) — MLX работает только на Apple Silicon/macOS,
  в Linux-контейнере их нет. Полный ансамбль из четырёх моделей в Docker не запускается;
  здесь доступен профиль CT2 (одна модель).
- **GigaAM (torch)** — можно добавить отдельным образом с `torch` CPU, но веса и лицензия
  требуют отдельного решения; сейчас не включён.
- **Веса моделей** — только том `./models` и манифест с sha256.

## Воспроизводимость

- версии Python-пакетов зафиксированы в `requirements-docker.txt`;
- `HF_HUB_OFFLINE=1` запрещает неявные загрузки;
- входное аудио и результат лежат в `./runs`, образ их не содержит;
- для полной идентичности используйте один и тот же образ (`asr-pipeline:0.1`) и тот же том модели.

## Ограничения

- Контейнер считает на CPU; на x86 будет медленнее, чем на Apple Silicon.
- `--engine whisper_ct2` — единственный движок в этом образе; `--ensemble all` здесь не сработает
  (нет MLX), это осознанное ограничение, а не ошибка конфигурации.

## Проверено на этом образе

- `docker build` → образ `asr-pipeline:0.1`, 1.79 ГБ;
- `docker run --rm asr-pipeline:0.1 bash scripts/smoke.sh` → `SMOKE PASS`;
- полный прогон на `audio.wav` (178 с) → `partial`, exit 2, все 10 стадий, `verify ok`;
- `verify` в отдельном контейнере без модели → 78 проверок, `ok`;
- `docker compose config -q` → валиден.

В `docker/engines.docker.json` два имени — `whisper_ct2` (основной) и `whisper` (алиас для стадии
repair, которая обращается по старому имени). Оба указывают на один и тот же CT2-адаптер.

## Linux-ансамбль (статус: не проверен в контейнере)

Файлы: `Dockerfile.linux`, `requirements-linux.txt`, `docker/engines.linux.json`,
`engines/linux_engine.py`, `docker/entrypoint.linux.sh`, `docker/prepare-models-linux.sh`.

Что подтверждено фактически:

- репозитории и лицензии: `ai-sage/GigaAM-v3` (MIT), `Systran/faster-whisper-large-v3` (MIT),
  `nvidia/parakeet-tdt-0.6b-v3` (CC-BY-4.0, есть `config.json` + safetensors для transformers),
  `Qwen/Qwen3-ASR-1.7B` (Apache-2.0, safetensors);
- GigaAM-веса скачаны и лежат в `models/linux/hf` (428 МБ), остальные — нет;
- `Dockerfile.linux` и `engines.linux.json` написаны, но **образ не собран и не прогнан**.

Чего не хватает для проверки:

1. Свободного места и рабочего Docker (сборка упёрлась в диск, containerd повредил blob — нужен перезапуск Docker Desktop).
2. Весов parakeet (~2.5 ГБ) и qwen (~3.4 ГБ): при 160 КБ/с это часы; через `HF_ENDPOINT=https://hf-mirror.com` — быстрее, но всё равно долго.

Поэтому в `engines.linux.json` сейчас `ensemble: true` только у `gigaam` и `whisper`,
у `parakeet` и `qwen` — `false` до загрузки весов. Включать их надо после `ONLY=nvidia/parakeet-tdt-0.6b-v3,Qwen/Qwen3-ASR-1.7B bash docker/prepare-models-linux.sh`.

Проверка, когда Docker поднимется:

```bash
bash docker/prepare-models-linux.sh          # или ONLY=… для части моделей
docker build -f Dockerfile.linux -t asr-pipeline:linux-0.1 .
docker run --rm -v "$PWD/models/linux:/models" asr-pipeline:linux-0.1 bash scripts/smoke.sh
docker run --rm -v "$PWD/models/linux:/models" -v "$PWD/runs:/runs" asr-pipeline:linux-0.1 \
  run --audio /runs/audio.wav --out /runs/out-linux \
  --ensemble all --engines /app/docker/engines.linux.json --vad energy --lid declared
```

До этого прогона Linux-ансамбль считается **непроверенным**: адаптеры на transformers
(GigaAM/Parakeet/Qwen) не запускались, pyannote для GigaAM установлен только в требованиях.
