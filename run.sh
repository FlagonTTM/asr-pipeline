#!/usr/bin/env bash
# Portable launcher. No installation, downloads or implicit ML preflight.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Ошибка: нет исполняемого %s. Подготовьте локальную .venv по README.md.\n' "$PYTHON" >&2
  exit 64
fi
export ASR_PY_ROOT="$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
args=()
help=0
for arg in "$@"; do
  case "$arg" in
    --online) unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE ;;
    --help|-h) help=1; args+=("$arg") ;;
    *) args+=("$arg") ;;
  esac
done
if [[ "$help" == 1 ]]; then
  printf '%s\n' 'run.sh: offline по умолчанию (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).' \
    '--online снимает эти ограничения; не устанавливает пакеты и не отменяет local_files_only адаптеров.'
  exec "$PYTHON" "$ROOT/asr_pipeline.py" run "${args[@]}"
fi
set -- ${args[@]+"${args[@]}"}
# Resolve JSON env templates here, not with eval or shell=True. Keep the
# resolved config in the job directory: verify/resume fingerprint this input.
"$PYTHON" - "$ROOT" "$@" <<'PY'
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(sys.argv[1])
args = sys.argv[2:]
pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

def expand(text):
    def replacement(match):
        name, default = match.groups()
        value = os.environ.get(name) or default
        if value is not None:
            if name == "ASR_X20_ROOT":
                value = str(Path(value).expanduser().resolve())
            os.environ[name] = value
            return value
        raise ValueError(f"не задана переменная {name}")
    return pattern.sub(replacement, text)

def option(name, default):
    value = default
    for index, arg in enumerate(args):
        if arg == name:
            if index + 1 >= len(args):
                raise ValueError(f"после {name} нужен аргумент")
            value = args[index + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value

try:
    source = Path(option("--engines", str(root / "engines.json"))).expanduser().resolve()
    config = json.loads(source.read_text(encoding="utf-8"))
    engines = config.get("engines", config)
    if not isinstance(engines, dict) or not engines:
        raise ValueError("engines.json должен содержать непустой объект движков")
    selected = option("--engine", os.environ.get("ASR_ENGINE") or config.get("primary") or next(iter(engines)))
    for name, cfg in engines.items():
        if not isinstance(cfg, dict) or "cmd" not in cfg:
            continue
        raw = cfg["cmd"]
        cfg["cmd"] = [expand(part) for part in raw] if isinstance(raw, list) else expand(raw)
    # The legacy repair planner requests the name 'whisper'. In explicit CT2
    # mode give it the same local adapter, so repairs never depend on x20.
    if selected == "whisper_ct2" and selected in engines:
        engines["whisper"] = dict(engines[selected])
        engines["whisper"]["ensemble"] = False
    if selected not in engines:
        raise ValueError(f"нет движка {selected!r} в {source}")
    cmd = engines[selected].get("cmd")
    if isinstance(cmd, list) and not option("--ensemble", ""):
        if not cmd or not Path(cmd[0]).is_file() or not os.access(cmd[0], os.X_OK):
            raise ValueError(f"нет интерпретатора {cmd[0] if cmd else '(пустой cmd)'}. "
                             "Проверьте ASR_X20_ROOT или выберите --engine whisper_ct2")
        if len(cmd) > 1 and cmd[1].endswith(".py") and not Path(cmd[1]).is_file():
            raise ValueError(f"нет файла движка {cmd[1]}")
    if os.environ.get("ASR_X20_ROOT"):
        # ASR_X20_ROOT is canonical; X20_ROOT is a compatibility output alias.
        os.environ["X20_ROOT"] = os.environ["ASR_X20_ROOT"]
    out = Path(option("--out", os.environ.get("ASR_OUT", str(Path.cwd() / "run")))).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rendered = out / ".engines.resolved.json"
    rendered.write_text(json.dumps(engines, ensure_ascii=False, indent=2), encoding="utf-8")
    defaults = ["--audio", os.environ.get("ASR_AUDIO", str(Path.cwd() / "audio.wav")),
                "--out", str(out), "--engine", selected, "--language", "ru", "--vad", "energy",
                "--lid", "declared", "--budget-frac", "0.35"]
    code = subprocess.run([sys.executable, str(root / "asr_pipeline.py"), "run",
                           *defaults, *args, "--engines", str(rendered)], check=False).returncode
except (OSError, ValueError, TypeError, KeyError) as exc:
    print(f"Ошибка запуска ASR: {exc}", file=sys.stderr)
    sys.exit(64)
sys.exit(code if code >= 0 else 1)
PY
