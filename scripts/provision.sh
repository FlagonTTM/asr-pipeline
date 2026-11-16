#!/usr/bin/env bash
# Explicit provisioning only; never invoked by run.sh or documentation checks.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
online=0
for arg in "$@"; do
  case "$arg" in
    --online) online=1 ;;
    -h|--help)
      printf '%s\n' 'Usage: bash scripts/provision.sh [--online]' \
        'Создаёт .venv (Python 3.12) и ставит requirements-lock.txt с проверкой хешей.' \
        'По умолчанию: только локальный Python и кэш uv, без сети и сборки sdist.' \
        '--online разрешает сеть для Python и пакетов, но не скачивает модели.'
      exit 0 ;;
    *) printf 'Ошибка: неизвестный аргумент %s (см. --help).\n' "$arg" >&2; exit 64 ;;
  esac
done
command -v uv >/dev/null 2>&1 || {
  printf 'Ошибка: uv не найден в PATH. Установите uv отдельно с разрешения оператора.\n' >&2
  exit 64
}
[[ -s "$ROOT/requirements-lock.txt" ]] || {
  printf 'Ошибка: отсутствует requirements-lock.txt в %s.\n' "$ROOT" >&2; exit 64;
}
cd -- "$ROOT"
network=(--offline --no-python-downloads)
export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never
if [[ "$online" == 1 ]]; then
  network=()
  export UV_OFFLINE=0 UV_PYTHON_DOWNLOADS=automatic
fi
if [[ -e .venv || -L .venv ]]; then
  if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c \
      'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    printf 'Ошибка: существующая .venv не является рабочей Python 3.12; не перезаписываю её. Переместите её вручную.\n' >&2
    exit 64
  fi
elif ! uv venv --python 3.12 .venv ${network[@]+"${network[@]}"}; then
  printf 'Ошибка создания .venv: нужен локальный Python 3.12. Сеть запрещена без явного --online.\n' >&2
  exit 1
fi
if ! uv pip install --python .venv/bin/python --require-hashes --only-binary :all: \
    --strict -r requirements-lock.txt ${network[@]+"${network[@]}"}; then
  printf '%s\n' 'Ошибка установки: проверьте локальный кэш uv, совместимость платформы и SHA-256.' \
    'Offline требует заранее подготовленных дистрибутивов. --online разрешает их загрузку.' \
    'Хеши не отключаются; модели и внешние x20 venv устанавливаются отдельно.' >&2
  exit 1
fi
printf 'PROVISION PASS: .venv Python 3.12, зависимости по requirements-lock.txt; модели не устанавливались.\n'
