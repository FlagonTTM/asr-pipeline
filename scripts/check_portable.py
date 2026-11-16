#!/usr/bin/env python3
"""Read-only portability preflight: Python literals, engine targets and docs.

No model imports, subprocesses, eval, downloads or interpreter execution.
Python files in tests are checked too; environments/caches/run outputs are not.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "node_modules", ".tmp"}
ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
# Concrete filesystem paths inside larger strings (including embedded code).
ABS_PATH = re.compile(r"(?<![\w:/])(?:[A-Za-z]:[\\/]|/(?!/))(?:[\w.~-]+[\\/])+[\w.~-]+")
ABS_START = re.compile(r"^(?:/[\w.~-]|[A-Za-z]:[\\/][\w.~-]|\\\\[^\\]+\\[^\\]+)")
# User-specific paths are never exempt, including tests. Synthetic /x, /abs,
# /srv/asr examples and platform paths are not personal-machine dependencies.
# Engine config ASR_X20_ROOT defaults remain the sole legacy config exception;
# x20_engine.py must resolve it from engines.json, never duplicate it in Python.
USER_PATH = re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\Users\\)[\w.~-]+(?:[/\\][\w.~-]+)*")
REQUIRED_SECTIONS = {
    "README.md": ("Что это", "Быстрый старт", "Артефакты", "Коды причин", "Exit-коды",
                  "Формат engines.json", "Свой движок", "Ограничения", "Лицензии моделей"),
    "INTEGRATION.md": ("Точки входа", "JSON-схемы", "Встраивание в оркестратор", "Чеклист приёмки"),
}
SCHEMAS = ("asr.segments/1", "asr.trust/2", "asr.route/1", "asr.repair.plan/1", "asr.summary/1", "asr.run/1")
FIELDS = {"cmd", "output", "model", "level_sensitive", "timeout_s", "ensemble", "family"}


def python_files(root):
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(name for name in names if name not in SKIP and not name.startswith("run-")
                          and name not in {"run", "output", "outputs", "models"})
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(directory) / name


def check_python(root):
    errors = []
    count = 0
    for path in python_files(root):
        count += 1
        label = str(path.relative_to(root))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=label)
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append(f"{label}: не разобран Python: {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value
            found = {match.group(0).strip() for match in USER_PATH.finditer(value)}
            for match in sorted(found):
                errors.append(f"{label}:{node.lineno}: путь пользователя в Python: {match}")
    return errors, count


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"повтор JSON-ключа {key!r}")
        result[key] = value
    return result


def expand(value, env):
    def replace(match):
        name, default = match.groups()
        current = env.get(name)
        if current:
            return current
        if default is not None:
            env[name] = default
            return default
        raise ValueError(f"не задана переменная {name}")
    return ENV.sub(replace, value)


def target_path(value, root):
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def check_engines(root, source, only=None):
    errors = []
    try:
        data = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        if not isinstance(data, dict):
            raise ValueError("верхний уровень должен быть объектом")
        engines = data.get("engines", data)
        if not isinstance(engines, dict) or not engines:
            raise ValueError("нет движков")
    except (OSError, UnicodeError, ValueError) as exc:
        return [f"{source.name}: {exc}"], 0
    expected = {"gigaam": "torch", "whisper": "mlx", "parakeet": "mlx", "qwen": "mlx", "whisper_ct2": "ct2"}
    for name, family in expected.items():
        cfg = engines.get(name)
        if not isinstance(cfg, dict):
            errors.append(f"engines.json: отсутствует объект движка {name}")
            continue
        missing = FIELDS - cfg.keys()
        if missing:
            errors.append(f"{name}: отсутствуют поля {sorted(missing)}")
        if cfg.get("family") != family:
            errors.append(f"{name}: family должен быть {family}")
        if cfg.get("ensemble") is not (name != "whisper_ct2"):
            errors.append(f"{name}: неверный флаг ensemble")
    env = dict(os.environ)
    env.setdefault("ASR_PY_ROOT", str(root))
    count = 0
    for name, cfg in engines.items():
        if not isinstance(cfg, dict):
            errors.append(f"{name}: описание должно быть объектом")
            continue
        if not isinstance(cfg.get("model"), str) or not cfg["model"]:
            errors.append(f"{name}: model должен быть непустой строкой")
        for key in ("level_sensitive", "ensemble"):
            if not isinstance(cfg.get(key), bool):
                errors.append(f"{name}: {key} должен быть bool")
        timeout = cfg.get("timeout_s")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            errors.append(f"{name}: timeout_s должен быть положительным конечным числом")
        output = cfg.get("output")
        if not isinstance(output, str) or not output or Path(output).is_absolute() or ".." in Path(output).parts:
            errors.append(f"{name}: output должен быть безопасным относительным путём")
        try:
            raw = cfg.get("cmd")
            parts = shlex.split(raw) if isinstance(raw, str) else raw
            if not isinstance(parts, list) or len(parts) < 2 or not all(isinstance(p, str) and p for p in parts):
                raise ValueError("cmd: ожидается interpreter + script + аргументы")
            # Absolute literals are allowed only in the X20 env default.
            for part in parts:
                scrubbed = ENV.sub("ENV", part)
                if any(m.group(2) and Path(m.group(2)).is_absolute() and m.group(1) != "ASR_X20_ROOT"
                       for m in ENV.finditer(part)) or ABS_PATH.search(scrubbed):
                    errors.append(f"{name}: абсолютный путь вне ASR_X20_ROOT default: {part}")
            resolved = [expand(part, env) for part in parts]
            if any("${" in part for part in resolved):
                raise ValueError("cmd содержит неразрешённую env-подстановку")
            if only is not None and name not in only:
                continue
            count += 1
            interpreter = target_path(resolved[0], root)
            if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                errors.append(f"{name}: нет исполняемого интерпретатора: {interpreter}")
            scripts = [part for part in resolved[1:] if part.endswith(".py")]
            if not scripts:
                errors.append(f"{name}: cmd не ссылается на Python-файл")
            for script in scripts:
                target = target_path(script, root)
                if not target.is_file():
                    errors.append(f"{name}: нет файла команды: {target}")
        except (ValueError, TypeError) as exc:
            errors.append(f"{name}: {exc}")
    if only is not None:
        for name in only - engines.keys():
            errors.append(f"неизвестный движок --engine {name}")
    return errors, count


def check_docs(root):
    errors = []
    for name, sections in REQUIRED_SECTIONS.items():
        try:
            text = (root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
        for section in sections:
            if not any(section in heading for heading in headings):
                errors.append(f"{name}: нет обязательного раздела {section!r}")
        if name == "INTEGRATION.md":
            for schema in SCHEMAS:
                if schema not in text:
                    errors.append(f"{name}: нет схемы {schema}")
            points = re.findall(r"^\d+\. \[ \] ", text, re.MULTILINE)
            if len(points) != 10:
                errors.append(f"{name}: ожидается 10 пунктов чеклиста, найдено {len(points)}")
            for phrase in ("verify пересчитывает покрытие", "summary не создаётся при failed"):
                if phrase not in text:
                    errors.append(f"{name}: нет критерия {phrase!r}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="корень проверяемого пакета")
    parser.add_argument("--engine", action="append", help="проверять наличие runtime только этих движков")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"нет каталога {root}")
    errors, python_count = check_python(root)
    engine_errors, engine_count = check_engines(root, root / "engines.json", set(args.engine) if args.engine else None)
    errors += engine_errors + check_docs(root)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"PORTABILITY: Python-файлов {python_count}, команд {engine_count}, ошибок {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
