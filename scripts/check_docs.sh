#!/usr/bin/env bash
# Read-only documentation contract. Only --help commands; never install or infer.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$#" == 1 && ( "$1" == --help || "$1" == -h ) ]]; then
  printf '%s\n' 'Usage: bash scripts/check_docs.sh' \
    'Проверяет разделы README/INTEGRATION, файлы и флаги по --help. Без сети/ML/установки.'
  exit 0
fi
if [[ "$#" != 0 ]]; then
  printf 'Ошибка: аргументы не поддерживаются (см. --help).\n' >&2; exit 64
fi
PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { printf 'Ошибка: сначала подготовьте .venv по README.md.\n' >&2; exit 64; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never
"$PYTHON" - "$ROOT" <<'PY'
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys

root = Path(sys.argv[1])
required = {
    "README.md": ("Что это", "Быстрый старт", "Provisioning и offline-режим",
                  "Проверка перед передачей", "Артефакты", "Коды причин", "Exit-коды",
                  "Формат engines.json", "Свой движок", "Ограничения", "Лицензии моделей"),
    "INTEGRATION.md": ("Точки входа", "Provisioning и offline-режим", "JSON-схемы",
                       "Встраивание в оркестратор", "Проверка перед передачей", "Чеклист приёмки"),
}
errors, docs = [], {}
for name, sections in required.items():
    try:
        text = (root / name).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"{name}: {exc}")
        continue
    docs[name] = text
    headings = set(re.findall(r"^#{1,6}\s+(.+)$", text, re.M))
    for section in sections:
        if section not in headings:
            errors.append(f"{name}: нет обязательного раздела {section!r}")
    for name_required in ("scripts/provision.sh", "scripts/check_docs.sh", "MODEL_MANIFEST.md", "LICENSES.md"):
        if name_required not in text:
            errors.append(f"{name}: нет ссылки на {name_required}")
    for command in (".venv/bin/python -m pytest -q", "bash scripts/smoke.sh",
                    ".venv/bin/python scripts/check_portable.py", "bash scripts/check_docs.sh",
                    ".venv/bin/python asr_pipeline.py verify --run ./run-final"):
        if command not in text:
            errors.append(f"{name}: нет команды handoff {command!r}")

# Generated outputs and explicitly illustrative/external inputs are not source
# files. Keep exceptions exact; never exclude every JSON, test or engine path.
artifacts = {
    ".engines.resolved.json", "manifest.json", "segments.json", "gigaam.segments.json",
    "whisper_ct2.segments.json", "trust.json", "trust.md", "vocab.md", "route.json",
    "plan.json", "additions.json", "segments_merged.json", "summary.json", "summary.md",
    "summary.rejected.md", "report.json", "run.json", "NAME.json", "receipt.json",
}
examples = {"my-engines.json", "engines/my_asr.py"}
files = re.compile(r"(?<![\w./-])(?:\./)?(?:[A-Za-z_][\w-]*/)*[A-Za-z_.][\w.-]*\.(?:py|sh|md|txt|json)(?![\w.-])")
refs = set()
for name, text in docs.items():
    for ref in set(files.findall(text)):
        ref = ref.removeprefix("./")
        # Consecutive artifact names are sometimes abbreviated with slashes.
        if all(part in artifacts for part in ref.split("/")) or ref in examples:
            continue
        if ref.split("/")[0] in {"00_input", "10_asr", "20_gate", "21_gate_after", "30_route", "40_repair", "50_summary", "60_verify"}:
            if Path(ref).name in artifacts:
                continue
        target = root / ref
        if not target.is_file() and "/" not in ref:
            # A prose basename is allowed only if it resolves to one shipped file.
            candidates = [root / folder / ref for folder in ("scripts", "engines")]
            found = [path for path in candidates if path.is_file()]
            target = found[0] if len(found) == 1 else target
        if not target.is_file():
            errors.append(f"{name}: нет упомянутого файла поставки {ref}")
        else:
            refs.add(str(target.relative_to(root)))

# Help is retrieved from real parsers. grep -F checks the mentioned long flags,
# with an exact-token regex after it to avoid accepting --outdated as --out.
commands = [[sys.executable, str(root / "asr_pipeline.py"), sub, "--help"]
            for sub in ("run", "gate", "route", "repair", "summarize", "verify", "status")]
commands += [[sys.executable, str(root / script), "--help"] for script in
             ("asr_gate.py", "asr_router.py", "asr_summary.py", "asr_ensemble.py",
              "engines/whisper_ct2.py", "engines/x20_engine.py", "scripts/check_portable.py")]
commands += [["bash", str(root / script), "--help"] for script in
             ("run.sh", "scripts/provision.sh", "scripts/check_docs.sh")]
if shutil.which("uv"):
    commands += [["uv", "venv", "--help"], ["uv", "pip", "install", "--help"]]
else:
    errors.append("uv отсутствует: невозможно сверить флаги provisioning по --help")
help_text = ""
for command in commands:
    try:
        result = subprocess.run(command, cwd=root, env=os.environ.copy(), capture_output=True,
                                text=True, timeout=30, check=False)
        if result.returncode:
            errors.append(f"--help завершился с exit {result.returncode}: {' '.join(command)}")
        else:
            help_text += result.stdout + "\n"
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"не удалось получить --help: {command}: {exc}")
flags = sorted(set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", "\n".join(docs.values()))))
for flag in flags:
    try:
        match = subprocess.run(["grep", "-F", "--", flag], input=help_text,
                               capture_output=True, text=True, check=False)
        if match.returncode or not re.search(re.escape(flag) + r"(?![\w-])", match.stdout):
            errors.append(f"флаг {flag} из документации не найден в --help")
    except OSError as exc:
        errors.append(f"grep недоступен: {exc}")
        break
for error in errors:
    print(f"ERROR: {error}", file=sys.stderr)
print(f"DOCS: документов {len(docs)}, файлов {len(refs)}, флагов {len(flags)}, ошибок {len(errors)}")
sys.exit(1 if errors else 0)
PY
