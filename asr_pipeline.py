#!/usr/bin/env python3
"""Orchestrator for the ASR pipeline: `asr.run/1` state, resume, verify.

Stages (each one is a subprocess of its own module, artifacts in their own dir):

    ingest -> 00_input/manifest.json          (ffprobe + sha256, no ML)
    asr    -> 10_asr/segments.json            (engine from engines.json)
    gate   -> 20_gate/trust.json              (asr_gate.py)
    route  -> 30_route/route.json             (asr_router.py)
    repair -> 40_repair/plan.json, additions  (asr_repair.py plan|run|merge)
    gate   -> 21_gate_after/trust.json        (asr_gate.py, after repair; attempts >1
                                               write 21_gate_after/a<N>/trust.json and
                                               run.json keeps the `final_trust` pointer)
    summary-> 50_summary/summary.md           (asr_summary.py)
    verify -> 60_verify/report.json           (invariants recomputed here)

State lives in `<out>/run.json`. `--resume` skips a stage only when its
`inputs_hash` (stage + params + content hashes of every input) matches the
previous run **and** all of its recorded outputs still exist.

Exit codes: 0 ok, 2 partial, 3 failed, 4 blocked, 1 internal error, 64 bad args.
`--legacy-exit` collapses 3 and 4 into 2.

Usage:
  asr_pipeline.py run --audio rec.wav --out run/ [--engines engines.json] \
                      [--peer kts=kts.json] [--resume] [--legacy-exit]
  asr_pipeline.py verify --run run/
  asr_pipeline.py status --run run/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_SCHEMA = "asr.run/1"
SEGMENTS_SCHEMA = "asr.segments/1"
TRUST_SCHEMA = "asr.trust/2"
REPORT_SCHEMA = "asr.run.report/1"
VERIFY_SCHEMA = "asr.verify/1"
DEFAULT_BUDGET_FRAC = 0.35

EXIT_OK, EXIT_ERROR, EXIT_PARTIAL, EXIT_FAILED, EXIT_BLOCKED, EXIT_ARGS = 0, 1, 2, 3, 4, 64
VERDICT_EXIT = {"ok": EXIT_OK, "partial": EXIT_PARTIAL, "failed": EXIT_FAILED, "blocked": EXIT_BLOCKED}
STAGE_RANK = {"ingest": 0, "asr": 1, "ensemble": 1.5, "gate": 2, "route": 3,
              "repair": 3.1, "repair_plan": 3.1, "repair_run": 3.2, "repair_merge": 3.3,
              "gate_after": 4, "summary": 5, "verify": 6}
OUTPUT_DIRS = ("00_input", "10_asr", "20_gate", "21_gate_after", "30_route",
               "40_repair", "50_summary", "60_verify", "logs")


class PipelineError(RuntimeError):
    """Stage/argument failure carrying a stable code and target exit code."""

    def __init__(self, code: str, detail: str, exit_code: int = EXIT_ERROR):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.exit_code = int(exit_code)


# --------------------------------------------------------------------------- utils

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def note(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_json(path) -> dict:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError("E_JSON", f"нет файла {p}") from exc
    except ValueError as exc:
        raise PipelineError("E_JSON", f"{p}: {exc}") from exc


def write_json(path, payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relpath(base: Path, path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(path)


def tail(text: str, limit: int = 800) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def inputs_hash(stage: str, params: dict, inputs: dict) -> str:
    """Content-addressed stage fingerprint: stage + params + every input's sha256."""
    payload = {
        "stage": stage,
        "params": params,
        "inputs": {str(key): (sha256_file(path) if path is not None and Path(path).is_file() else "MISSING")
                   for key, path in sorted(inputs.items())},
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))[:32]


def probe_audio(path: Path) -> dict:
    """Duration/sr/channels via ffprobe, falling back to soundfile (never ML)."""
    try:
        cp = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "format=duration:stream=sample_rate,channels", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        if cp.returncode == 0 and cp.stdout.strip():
            data = json.loads(cp.stdout)
            stream = (data.get("streams") or [{}])[0]
            duration = float((data.get("format") or {}).get("duration") or 0.0)
            if stream.get("sample_rate") and stream.get("channels") and duration > 0:
                return {"duration_s": round(duration, 2), "sr": int(stream["sample_rate"]),
                        "channels": int(stream["channels"]), "probe": "ffprobe"}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
        pass
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {"duration_s": round(float(info.duration), 2), "sr": int(info.samplerate),
                "channels": int(info.channels), "probe": "soundfile"}
    except Exception as exc:  # soundfile is optional for ingest
        raise PipelineError("E_AUDIO", f"не удалось прочитать аудио {path}: {exc}") from exc


def write_log(path: Path, cmd: list[str], completed) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ["$ " + " ".join(shlex.quote(str(part)) for part in cmd),
            f"exit: {completed.returncode}", "--- stdout ---", completed.stdout or "",
            "--- stderr ---", completed.stderr or ""]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def run_process(cmd, *, label: str, run_dir: Path | None = None, log: Path | None = None,
                allowed=(0,), timeout: float | None = None):
    env = os.environ.copy()
    env["ASR_STAGE"] = label
    if run_dir is not None:
        env["ASR_RUN_DIR"] = str(run_dir)
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout,
                                   cwd=str(run_dir or HERE))
    except FileNotFoundError as exc:
        raise PipelineError("E_SPAWN", f"{label}: не найден исполняемый файл {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError("E_TIMEOUT", f"{label}: таймаут {timeout} с") from exc
    if log is not None:
        write_log(log, cmd, completed)
    if allowed is not None and completed.returncode not in allowed:
        raise PipelineError("E_STAGE_FAILED",
                            f"{label} exit {completed.returncode}: {tail(completed.stderr or completed.stdout)}")
    return completed


def run_module(stages_dir: Path, module: str, argv, *, label: str, run_dir: Path | None = None,
               allowed=(0,), timeout: float | None = None):
    path = Path(stages_dir) / module
    if not path.exists():
        raise PipelineError("E_MODULE_MISSING",
                            f"нет модуля {path} (стадия {label}) — модуль ещё не реализован")
    log = (Path(run_dir) / "logs" / f"{label}.log") if run_dir is not None else None
    return run_process([sys.executable, str(path), *argv], label=label, run_dir=run_dir,
                       log=log, allowed=allowed, timeout=timeout)


ENV_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """${VAR} and ${VAR:-default} in engine commands; no shell is involved."""
    def repl(match):
        name, default = match.group(1), match.group(2)
        if name == "ASR_PY_ROOT" and name not in os.environ:
            return str(Path(__file__).resolve().parent)
        return os.environ.get(name, default if default is not None else "")
    return ENV_PLACEHOLDER.sub(repl, text)


def substitute_cmd(cmd, audio: Path, out: Path, language: str) -> list[str]:
    if cmd is None:
        return []
    parts = shlex.split(cmd) if isinstance(cmd, str) else [str(part) for part in cmd]
    parts = [expand_env(part) for part in parts]
    values = {"audio": str(audio), "out": str(out), "language": str(language)}
    resolved = []
    for part in parts:
        try:
            resolved.append(part.format(**values))
        except (KeyError, IndexError, ValueError):
            resolved.append(part)
    return resolved


def find_transcript(directory: Path, configured=None) -> Path:
    if configured:
        candidate = Path(configured)
        candidate = candidate if candidate.is_absolute() else directory / candidate
        if candidate.exists():
            return candidate
    candidate = directory / "segments.json"
    if candidate.exists():
        return candidate
    found = sorted(p for p in directory.glob("*.json") if p.is_file())
    if not found:
        raise PipelineError("E_ENGINE_OUTPUT", f"движок не создал ни одного JSON в {directory}")
    found.sort(key=lambda p: p.stat().st_size, reverse=True)
    return found[0]


def adopt_json(directory: Path, target_name: str, *, schema_hint: str | None = None,
               exclude: set[str] | None = None) -> Path:
    """Rename/copy the best JSON candidate in `directory` to `target_name`."""
    directory = Path(directory)
    target = directory / target_name
    if target.exists():
        return target
    excluded = set(exclude or ())
    excluded.add(target_name)
    candidates = [p for p in sorted(directory.glob("*.json")) if p.name not in excluded and p.is_file()]
    if not candidates:
        raise PipelineError("E_STAGE_OUTPUTS", f"в {directory} нет JSON для {target_name}")
    chosen = None
    if schema_hint:
        for path in candidates:
            try:
                if str(json.loads(path.read_text(encoding="utf-8")).get("schema") or "") == schema_hint:
                    chosen = path
                    break
            except (OSError, ValueError):
                continue
    if chosen is None:
        chosen = max(candidates, key=lambda p: p.stat().st_size)
    shutil.copyfile(chosen, target)
    return target


def load_engines(path=None, primary: str | None = None):
    source = (Path(path).expanduser() if path else HERE / "engines.json").resolve()
    if not source.exists():
        raise PipelineError("E_ENGINES", f"нет файла движков {source}", EXIT_ARGS)
    data = read_json(source)
    raw = data.get("engines") if isinstance(data.get("engines"), dict) else data
    engines = {}
    for key, value in raw.items():
        if isinstance(value, dict) and ("cmd" in value or "command" in value):
            cfg = dict(value)
            cfg.setdefault("cmd", cfg.get("command"))
            engines[key] = cfg
    if not engines:
        raise PipelineError("E_ENGINES", f"{source}: нет описаний движков", EXIT_ARGS)
    name = primary or data.get("primary") or next(iter(engines))
    if name not in engines:
        raise PipelineError("E_ENGINES", f"{source}: нет движка {name!r} (есть {sorted(engines)})", EXIT_ARGS)
    return engines, name, source


def parse_peers(specs) -> dict:
    peers = {}
    for spec in specs or []:
        name, sep, path = str(spec).partition("=")
        if not sep or not name.strip() or not path.strip():
            raise PipelineError("E_ARGS", f"--peer ожидает NAME=PATH, получено {spec!r}", EXIT_ARGS)
        peers[name.strip()] = Path(path).expanduser().resolve()
    return peers


def route_blocked(route: dict | None) -> bool:
    if not isinstance(route, dict):
        return False
    if route.get("blocked"):
        return True
    return str(route.get("verdict") or "") == "blocked"


def uncovered_s(trust: dict | None) -> float:
    if not isinstance(trust, dict):
        return 0.0
    total = 0.0
    for interval in trust.get("uncovered_all") or []:
        if isinstance(interval, dict):
            try:
                total += max(0.0, float(interval["end"]) - float(interval["start"]))
            except (KeyError, TypeError, ValueError):
                continue
    return round(total, 1)


def union_length(segments) -> float:
    """Union length of segment spans (the only coverage the contract counts)."""
    spans: list[list[float]] = []
    for row in sorted(segments or [], key=lambda r: float(r.get("start") or 0.0)):
        if not isinstance(row, dict):
            continue
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        if spans and start <= spans[-1][1] + 1e-9:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return round(sum(end - start for start, end in spans), 3)


# --------------------------------------------------------------------------- state

class RunState:
    """`run.json` (asr.run/1): stage ledger + outcome, saved after every stage."""

    def __init__(self, run_dir: Path, resume: bool):
        self.dir = Path(run_dir)
        self.path = self.dir / "run.json"
        self.resume = bool(resume)
        self.prev: dict[str, dict] = {}
        self.data = {
            "schema": RUN_SCHEMA,
            "run_dir": str(self.dir),
            "created_at": now_iso(),
            "stages": [],
            "attempts": {"prep": 0, "repair": 0},
            "engines": {},
            "verdict": None,
            "exit_code": None,
            "unresolved_s": None,
        }
        if not self.path.exists():
            return
        try:
            previous = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        if isinstance(previous, dict) and previous.get("schema") == RUN_SCHEMA:
            self.prev = {s["name"]: s for s in previous.get("stages") or []
                         if isinstance(s, dict) and s.get("name")}
            self.data["created_at"] = previous.get("created_at") or self.data["created_at"]
            self.data["attempts"] = previous.get("attempts") or self.data["attempts"]
            if previous.get("final_trust"):
                self.data["final_trust"] = previous["final_trust"]
            if previous.get("effective_transcript"):
                self.data["effective_transcript"] = previous["effective_transcript"]
        elif isinstance(previous, dict):
            self.data.setdefault("notes", []).append(
                "предыдущий run.json с другой схемой — начат новый прогон")

    def record(self, name: str, status: str, inputs_digest: str | None = None, outputs=None,
               exit_code: int = EXIT_OK, duration_s: float | None = None, reason: str | None = None,
               error: str | None = None, extra: dict | None = None) -> dict:
        record = {"name": name, "status": status}
        if inputs_digest is not None:
            record["inputs_hash"] = inputs_digest
        record["outputs"] = list(outputs or [])
        record["exit"] = int(exit_code)
        if duration_s is not None:
            record["duration_s"] = duration_s
        if reason:
            record["reason"] = reason
        if error:
            record["error"] = error
        if extra:
            record.update(extra)
        self.data["stages"] = [s for s in self.data["stages"] if s.get("name") != name]
        self.data["stages"].append(record)
        return record

    def save(self) -> None:
        write_json(self.path, self.data)


# ------------------------------------------------------------------------ pipeline

class Pipeline:
    def __init__(self, audio: Path | None, out: Path, *, engines=None, primary: str | None = None,
                 peer=None, language: str = "ru", vad: str = "silero", lid: str = "whisper",
                 budget_frac: float = DEFAULT_BUDGET_FRAC, resume: bool = False,
                 stages_dir=None, legacy_exit: bool = False, ensemble: str | None = None):
        self.out = Path(out).expanduser().resolve()
        self.audio = Path(audio).expanduser().resolve() if audio else None
        self.language = language
        self.vad = vad
        self.lid = lid
        self.budget_frac = float(budget_frac)
        self.resume = bool(resume)
        self.legacy_exit = bool(legacy_exit)
        self.stages_dir = Path(stages_dir).expanduser().resolve() if stages_dir \
            else Path(os.environ.get("ASR_STAGES_DIR") or HERE).resolve()
        self.engines, self.primary, self.engines_path = load_engines(engines, primary)
        self.decoder_primary = self.primary
        self.ensemble_names = []
        if ensemble is not None:
            requested = ([name for name, cfg in self.engines.items() if cfg.get("ensemble") is True]
                         if ensemble == "all" else [name.strip() for name in ensemble.split(",")])
            for name in requested:
                if name not in self.engines or (name != self.primary and self.engines[name].get("ensemble") is not True):
                    raise PipelineError("E_ENSEMBLE", f"движок {name!r} неизвестен или ensemble!=true", EXIT_ARGS)
            self.ensemble_names = list(dict.fromkeys([self.primary, *requested]))
            for name in self.ensemble_names:
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {"segments", "report", "primary_attempt"}:
                    raise PipelineError("E_ENSEMBLE", f"небезопасное имя движка {name!r}", EXIT_ARGS)
        self.peers = parse_peers(peer)
        self.state = RunState(self.out, self.resume)
        self.state.data["engines_sha256"] = sha256_file(self.engines_path)
        self.exit_code = EXIT_ERROR
        self._regate_done = False
        self.summary_refused = False

    # -- paths -------------------------------------------------------------
    def p(self, *parts) -> Path:
        return self.out.joinpath(*parts)

    @property
    def manifest(self) -> Path:
        return self.p("00_input", "manifest.json")

    @property
    def engines_copy(self) -> Path:
        return self.p("00_input", "engines.json")

    @property
    def segments(self) -> Path:
        return self.p("10_asr", "segments.json")

    @property
    def trust(self) -> Path:
        return self.p("20_gate", "trust.json")

    @property
    def route(self) -> Path:
        return self.p("30_route", "route.json")

    @property
    def plan(self) -> Path:
        return self.p("40_repair", "plan.json")

    @property
    def additions(self) -> Path:
        return self.p("40_repair", "additions.json")

    def merged_path(self, attempt: int) -> Path:
        if attempt and attempt > 1:
            return self.p("40_repair", f"segments_merged_a{attempt}.json")
        return self.p("40_repair", "segments_merged.json")

    def gate_dir(self, attempt: int) -> Path:
        return self.p("21_gate_after") if attempt <= 1 else self.p("21_gate_after", f"a{attempt}")

    def trust_after_path(self, attempt: int) -> Path:
        return self.gate_dir(attempt) / "trust.json"

    @property
    def trust_after(self) -> Path:
        attempt = int((self.state.data.get("attempts") or {}).get("repair") or 1)
        return self.trust_after_path(attempt)

    def effective(self) -> Path:
        attempt = int((self.state.data.get("attempts") or {}).get("repair") or 0)
        return self.merged_path(attempt) if attempt else self.segments

    def final_trust(self) -> Path:
        recorded = self.state.data.get("final_trust")
        if recorded:
            candidate = self.out / recorded
            if candidate.exists():
                return candidate
        if self.trust_after.exists():
            return self.trust_after
        return self.trust

    def route_doc(self) -> dict:
        return read_json(self.route) if self.route.exists() else {}

    def engines_record(self) -> dict:
        return {name: {"cmd": cfg.get("cmd"), "model": cfg.get("model"),
                       "level_sensitive": bool(cfg.get("level_sensitive", False))}
                for name, cfg in self.engines.items()}

    # -- stage plumbing ----------------------------------------------------
    def run_stage(self, name: str, inputs: dict, params: dict, outputs: list[Path], action, extra=None,
                  outputs_check=None) -> str:
        digest = inputs_hash(name, params, inputs)
        extra = {**(extra or {}), "fingerprint": {"params": params, "inputs": {
            key: relpath(self.out, path) if path is not None else None for key, path in inputs.items()}}}
        rel_outputs = [relpath(self.out, path) for path in outputs]

        def ready() -> bool:
            if outputs_check is not None:
                return bool(outputs_check())
            return all(path.exists() for path in outputs)

        previous = self.state.prev.get(name)
        if (self.resume and previous and previous.get("inputs_hash") == digest
                and previous.get("status") in {"done", "skipped"} and ready()
                and all((self.out / rel).is_file() for rel in previous.get("outputs") or [])):
            self.state.record(name, "skipped", digest, rel_outputs, int(previous.get("exit") or 0),
                              reason="inputs unchanged", extra=extra)
            self.state.save()
            note(f"[{name}] skip (inputs unchanged)")
            return "skipped"
        started = time.monotonic()
        try:
            code = action()
        except PipelineError as exc:
            self.state.record(name, "failed", digest, rel_outputs, exc.exit_code, error=str(exc), extra=extra)
            self.state.save()
            raise
        if not ready():
            missing = [relpath(self.out, path) for path in outputs if not path.exists()]
            self.state.record(name, "failed", digest, rel_outputs, EXIT_ERROR,
                              error=f"нет выходов: {missing}", extra=extra)
            self.state.save()
            raise PipelineError("E_STAGE_OUTPUTS", f"стадия {name} не создала {missing}")
        self.state.record(name, "done", digest, rel_outputs, int(code or 0),
                          duration_s=round(time.monotonic() - started, 3), extra=extra)
        self.state.save()
        note(f"[{name}] done")
        return "done"

    def record_skip(self, name: str, reason: str) -> None:
        params = {"skip_reason": reason}
        self.state.record(name, "skipped", inputs_hash(name, params, {}), [], EXIT_OK, reason=reason,
                          extra={"fingerprint": {"params": params, "inputs": {}}})
        self.state.save()
        note(f"[{name}] skip ({reason})")

    def call_module(self, module: str, argv, label: str, allowed=(0,), timeout=None) -> int:
        completed = run_module(self.stages_dir, module, argv, label=label, run_dir=self.out,
                               allowed=allowed, timeout=timeout)
        return completed.returncode

    def reset_outputs(self) -> None:
        for name in OUTPUT_DIRS:
            target = self.out / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
        note("[run] выходные каталоги очищены (без --resume)")

    # -- stages ------------------------------------------------------------
    def stage_ingest(self) -> str:
        def action():
            info = probe_audio(self.audio)
            manifest = {
                "schema": "asr.input/1",
                "created_at": now_iso(),
                "audio": {"path": str(self.audio), "name": self.audio.name,
                          "sha256": sha256_file(self.audio), "size_bytes": self.audio.stat().st_size,
                          "duration_s": info["duration_s"], "sr": info["sr"], "channels": info["channels"]},
                "probe": info["probe"],
                "language": self.language,
            }
            write_json(self.manifest, manifest)
            # flat name->cfg map: this is the shape asr_repair.py reads (--engines)
            write_json(self.engines_copy, self.engines)
            return EXIT_OK

        return self.run_stage("ingest", {"audio": self.audio, "engines": self.engines_path},
                              {"audio": str(self.audio), "language": self.language},
                              [self.manifest, self.engines_copy], action)

    def stage_asr(self) -> str:
        cfg = self.engines[self.primary]
        params = {"engine": self.primary, "cmd": cfg.get("cmd"), "model": cfg.get("model"),
                  "language": self.language, "ensemble": self.ensemble_names}
        target = self.p("10_asr", f"{self.primary}.json") if self.ensemble_names else self.segments
        attempt_path = self.p("10_asr", "primary_attempt.json")

        def decode():
            manifest = read_json(self.manifest)
            raw_dir = self.p("10_asr", "engine")
            if raw_dir.is_dir():
                shutil.rmtree(raw_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)
            cmd = substitute_cmd(cfg.get("cmd"), self.audio, raw_dir, self.language)
            if not cmd:
                raise PipelineError("E_ENGINE", f"движок {self.primary}: пустой cmd")
            completed = run_process(cmd, label="asr", run_dir=self.out,
                                    log=self.p("logs", "asr.log"),
                                    allowed=(0,), timeout=float(cfg.get("timeout_s") or 900))
            produced = find_transcript(raw_dir, cfg.get("output"))
            self._write_segments(produced, manifest, cfg, target=target)
            return completed.returncode

        def action():
            if not self.ensemble_names:
                return decode()
            target.unlink(missing_ok=True)
            try:
                code = decode()
            except (PipelineError, OSError, ValueError, TypeError) as exc:
                target.unlink(missing_ok=True)
                write_json(attempt_path, {"ok": False, "engine": self.decoder_primary,
                                         "note": f"ensemble: {self.decoder_primary} исключён: {exc}"})
                return EXIT_OK  # completed attempt; peers can still produce a transcript
            write_json(attempt_path, {"ok": True, "engine": self.decoder_primary})
            return code

        status = self.run_stage("asr", {"manifest": self.manifest, "engines": self.engines_copy},
                                params, [attempt_path] if self.ensemble_names else [target], action)
        if self.ensemble_names and read_json(attempt_path).get("ok"):
            stage = next(s for s in self.state.data["stages"] if s["name"] == "asr")
            stage["outputs"].append(relpath(self.out, target))
            self.state.save()
        return status

    def _write_segments(self, produced: Path, manifest: dict, cfg: dict, *,
                        target: Path | None = None, engine_name: str | None = None) -> None:
        target = target or self.segments
        engine_name = engine_name or self.primary
        data = read_json(produced)
        if isinstance(data, dict) and str(data.get("schema") or "").startswith("asr.segments"):
            document = data
        else:
            audio_info = manifest["audio"]
            argv = ["--in", str(produced), "--out", str(target), "--audio", str(self.audio),
                    "--duration", str(audio_info["duration_s"]), "--sr", str(audio_info["sr"]),
                    "--channels", str(audio_info["channels"]), "--engine", engine_name,
                    "--language", self.language]
            if cfg.get("model"):
                argv += ["--model", str(cfg["model"])]
            if cfg.get("version"):
                argv += ["--version", str(cfg["version"])]
            if cfg.get("level_sensitive"):
                argv += ["--level-sensitive"]
            self.call_module("adapt_legacy.py", argv, "asr")
            document = read_json(target)

        audio_info = manifest["audio"]
        document["schema"] = SEGMENTS_SCHEMA
        document["audio"] = {"path": str(self.audio), "sha256": audio_info["sha256"],
                             "duration_s": audio_info["duration_s"], "sr": audio_info["sr"],
                             "channels": audio_info["channels"]}
        document["engine"] = {"name": engine_name, "model": cfg.get("model"),
                              "version": cfg.get("version"), "params": dict(cfg.get("params") or {}),
                              "level_sensitive": bool(cfg.get("level_sensitive", False))}
        rows = []
        for index, row in enumerate(document.get("segments") or [], 1):
            if not isinstance(row, dict):
                continue
            row = dict(row)
            row.setdefault("id", f"s{index:04d}")
            row.setdefault("speaker", None)
            row.setdefault("conf", None)
            row.setdefault("source", "primary")
            row.setdefault("flags", [])
            rows.append(row)
        if not rows:
            raise PipelineError("E_ASR_EMPTY", f"движок {self.primary} вернул пустой транскрипт")
        document["segments"] = rows
        document.setdefault("meta", {"vad": None, "window_s": None})
        write_json(target, document)

    def stage_ensemble(self) -> str:
        """Decode eligible peers, then select medoid segments via the CLI."""
        report_path = self.p("10_asr", "ensemble", "report.json")
        primary_path = self.p("10_asr", f"{self.decoder_primary}.json")
        attempt_path = self.p("10_asr", "primary_attempt.json")
        primary_attempt = read_json(attempt_path)

        def action():
            manifest = read_json(self.manifest)
            successful = [self.decoder_primary] if primary_attempt.get("ok") else []
            notes = [primary_attempt["note"]] if primary_attempt.get("note") else []
            for name in self.ensemble_names:
                if name == self.decoder_primary:
                    continue
                cfg = self.engines[name]
                raw = self.p("10_asr", "engines", name)
                target = self.p("10_asr", f"{name}.json")
                if raw.is_dir():
                    shutil.rmtree(raw)
                raw.mkdir(parents=True, exist_ok=True)
                target.unlink(missing_ok=True)
                try:
                    cmd = substitute_cmd(cfg.get("cmd"), self.audio, raw, self.language)
                    if not cmd:
                        raise PipelineError("E_ENGINE", f"{name}: пустой cmd")
                    run_process(cmd, label=f"ensemble_{name}", run_dir=self.out,
                                log=self.p("logs", f"ensemble_{name}.log"),
                                timeout=float(cfg.get("timeout_s") or 900))
                    self._write_segments(find_transcript(raw, cfg.get("output")), manifest, cfg,
                                         target=target, engine_name=name)
                    successful.append(name)
                except (PipelineError, OSError, ValueError, TypeError) as exc:
                    target.unlink(missing_ok=True)
                    notes.append(f"ensemble: {name} исключён: {exc}")
            enabled = len(successful) >= 2
            if enabled:
                argv = []
                for name in successful:
                    argv += ["--segments", f"{name}={self.p('10_asr', name + '.json')}"]
                argv += ["--out", str(report_path.parent), "--audio", str(self.audio)]
                self.call_module("asr_ensemble.py", argv, "ensemble")
                document = read_json(report_path.parent / "segments.json")
                # Ensemble CLI intentionally leaves sha256 empty; bind the result
                # to the same ingested audio as every single-engine transcript.
                document["audio"] = {k: manifest["audio"][k]
                                     for k in ("path", "sha256", "duration_s", "sr", "channels")}
                if not document.get("segments"):
                    raise PipelineError("E_ASR_EMPTY", "ансамбль вернул пустой транскрипт")
                write_json(self.segments, document)
                report = read_json(report_path)
            else:
                if not successful:
                    raise PipelineError("E_ASR_EMPTY", "все движки ансамбля завершились без транскрипта")
                shutil.copyfile(self.p("10_asr", successful[0] + ".json"), self.segments)
                report = {"schema": "asr.ensemble/1", "votes": {}, "mean_confidence": None}
                notes.append("ensemble: осталось менее 2 движков — используется основной транскрипт")
            report.update({"engines": successful, "n_votes": len(successful), "enabled": enabled, "notes": notes})
            write_json(report_path, report)
            return EXIT_OK

        inputs = {"primary_attempt": attempt_path, "engines": self.engines_copy, "audio": self.audio}
        if primary_attempt.get("ok"):
            inputs["primary"] = primary_path
        status = self.run_stage("ensemble", inputs, {"engines": self.ensemble_names},
                                [self.segments, report_path], action)
        report = read_json(report_path)
        self.state.data["ensemble"] = {key: report.get(key) for key in
                                       ("engines", "votes", "mean_confidence", "n_votes", "enabled")}
        self.primary = "ensemble" if report.get("enabled") else report["engines"][0]
        for message in report.get("notes") or []:
            if message not in self.state.data.setdefault("notes", []):
                self.state.data["notes"].append(message)
        stage = next(s for s in self.state.data["stages"] if s["name"] == "ensemble")
        stage["outputs"] += [f"10_asr/{name}.json" for name in report["engines"]]
        self.state.save()
        return status

    def _invalidate_downstream(self) -> None:
        self.state.data["attempts"] = {"prep": 0, "repair": 0}
        for key in ("final_trust", "effective_transcript"):
            self.state.data.pop(key, None)
        for name in list(self.state.prev):
            if STAGE_RANK.get(name, -1) >= STAGE_RANK["gate"]:
                del self.state.prev[name]
        for directory in ("20_gate", "21_gate_after", "30_route", "40_repair", "50_summary", "60_verify"):
            target = self.p(directory)
            if target.is_dir():
                shutil.rmtree(target)
        self.state.save()

    def _gate_argv(self, transcript: Path, out_dir: Path) -> list[str]:
        argv = ["--audio", str(self.audio), "--transcript", f"{self.primary}={transcript}"]
        for name, path in self.peers.items():
            argv += ["--transcript", f"{name}={path}"]
        argv += ["--primary", self.primary, "--vad", self.vad, "--lid", self.lid,
                 "--language", self.language, "--out", str(out_dir)]
        return argv

    def stage_gate(self) -> str:
        params = {"primary": self.primary, "vad": self.vad, "lid": self.lid, "language": self.language}
        inputs = {"segments": self.segments, **{f"peer:{name}": path for name, path in self.peers.items()}}

        def action():
            return self.call_module("asr_gate.py", self._gate_argv(self.segments, self.p("20_gate")),
                                    "gate", allowed=(0, 2))

        return self.run_stage("gate", inputs, params, [self.trust], action)

    def stage_route(self) -> str:
        def action():
            argv = ["--trust", str(self.trust), "--out", str(self.route),
                    "--budget-frac", str(self.budget_frac)]
            return self.call_module("asr_router.py", argv, "route", allowed=(0, 2, 3, 4))

        return self.run_stage("route", {"trust": self.trust}, {"budget_frac": self.budget_frac},
                              [self.route], action)

    def plan_trust(self, attempt: int) -> Path:
        return self.trust if attempt == 1 else self.trust_after_path(attempt - 1)

    def stage_repair_plan(self, attempt: int) -> dict:
        trust_path = self.plan_trust(attempt)
        params = {"attempt": attempt, "trust": relpath(self.out, trust_path)}
        inputs = {"trust": trust_path, "route": self.route}

        def action():
            plan_dir = self.p("40_repair")
            plan_dir.mkdir(parents=True, exist_ok=True)
            argv = ["plan", "--trust", str(trust_path), "--route", str(self.route), "--out", str(plan_dir)]
            code = self.call_module("asr_repair.py", argv, "repair_plan", allowed=(0, 2))
            if not self.plan.exists():
                adopt_json(plan_dir, "plan.json", schema_hint="asr.repair.plan/1")
            return code

        self.run_stage("repair_plan", inputs, params, [self.plan], action)
        return read_json(self.plan)

    def stage_repair_run(self, attempt: int) -> str:
        params = {"attempt": attempt, "plan": relpath(self.out, self.plan)}

        def action():
            repair_dir = self.p("40_repair")
            argv = ["run", "--audio", str(self.audio), "--plan", str(self.plan),
                    "--engines", str(self.engines_copy), "--out", str(repair_dir)]
            code = self.call_module("asr_repair.py", argv, "repair_run", allowed=(0, 2))
            if not self.additions.exists():
                adopt_json(repair_dir, "additions.json",
                           exclude={self.plan.name, self.merged_path(attempt).name})
            return code

        return self.run_stage("repair_run", {"plan": self.plan, "audio": self.audio,
                                             "engines": self.engines_copy}, params,
                              [self.additions], action)

    def stage_repair_merge(self, attempt: int) -> str:
        # deterministic bases keep resume stable across attempts
        base = self.segments if attempt == 1 else self.merged_path(attempt - 1)
        target = self.merged_path(attempt)

        def action():
            argv = ["merge", "--base", str(base), "--additions", str(self.additions), "--out", str(target)]
            return self.call_module("asr_repair.py", argv, "repair_merge", allowed=(0, 2))

        status = self.run_stage("repair_merge", {"base": base, "additions": self.additions},
                                {"attempt": attempt}, [target], action)
        self.state.data["attempts"]["repair"] = attempt
        self.state.data["effective_transcript"] = relpath(self.out, target)
        self.state.save()
        return status

    def stage_gate_after(self, attempt: int) -> str:
        params = {"primary": self.primary, "vad": self.vad, "lid": self.lid,
                  "language": self.language, "attempt": attempt}
        transcript = self.merged_path(attempt)
        target = self.trust_after_path(attempt)
        inputs = {"segments": transcript,
                  **{f"peer:{name}": path for name, path in self.peers.items()}}

        def action():
            return self.call_module("asr_gate.py", self._gate_argv(transcript, self.gate_dir(attempt)),
                                    "gate_after", allowed=(0, 2))

        status = self.run_stage("gate_after", inputs, params, [target], action)
        self.state.data["final_trust"] = relpath(self.out, target)
        self.state.save()
        self._regate_done = True
        return status

    def _repair_phase_done(self) -> bool:
        """True when the recorded final attempt still matches the artifacts on disk."""
        attempt = int((self.state.data.get("attempts") or {}).get("repair") or 0)
        gate_path = self.trust_after_path(attempt) if attempt else None
        if not attempt or gate_path is None or not gate_path.exists() or not self.merged_path(attempt).exists():
            return False
        if self.final_trust().resolve() != gate_path.resolve():
            return False
        plan_trust = self.plan_trust(attempt)
        base = self.segments if attempt == 1 else self.merged_path(attempt - 1)
        gate_params = {"primary": self.primary, "vad": self.vad, "lid": self.lid,
                       "language": self.language, "attempt": attempt}
        specs = {
            "repair_plan": ({"attempt": attempt, "trust": relpath(self.out, plan_trust)},
                            {"trust": plan_trust, "route": self.route}),
            "repair_run": ({"attempt": attempt, "plan": relpath(self.out, self.plan)},
                           {"plan": self.plan, "audio": self.audio, "engines": self.engines_copy}),
            "repair_merge": ({"attempt": attempt}, {"base": base, "additions": self.additions}),
            "gate_after": (gate_params, {"segments": self.merged_path(attempt),
                                         **{f"peer:{name}": path for name, path in self.peers.items()}}),
        }
        for name, (params, inputs) in specs.items():
            prev = self.state.prev.get(name) or {}
            outputs = prev.get("outputs")
            if (prev.get("status") not in {"done", "skipped"} or not isinstance(outputs, list)
                    or not outputs or any(not isinstance(rel, str) or not (self.out / rel).is_file()
                                          for rel in outputs)):
                return False
            fingerprint = {"params": params, "inputs": {
                key: relpath(self.out, path) if path is not None else None for key, path in inputs.items()}}
            if (prev.get("fingerprint") != fingerprint
                    or any(path is None or not Path(path).is_file() for path in inputs.values())
                    or prev.get("inputs_hash") != inputs_hash(name, params, inputs)):
                return False
        return True

    def stage_repair(self) -> None:
        route = self.route_doc()
        steps = [s for s in route.get("steps") or [] if isinstance(s, dict)]
        if not any(str(step.get("kind")) == "repair" for step in steps):
            self.record_skip("repair", "no repair step in route")
            return
        if self.resume and self._repair_phase_done():
            for name in ("repair_plan", "repair_run", "repair_merge", "gate_after"):
                prev = self.state.prev.get(name) or {}
                self.state.record(name, "skipped", prev.get("inputs_hash"), prev.get("outputs"),
                                  int(prev.get("exit") or 0), reason="inputs unchanged",
                                  extra={"phase": "repair", "fingerprint": prev.get("fingerprint")})
            self.state.save()
            note("[repair] skip (repair phase already complete)")
            self._regate_done = True
            return
        tries = max(1, min(int((route.get("budget") or {}).get("attempts_max") or 1), 3))
        for attempt in range(1, tries + 1):
            plan = self.stage_repair_plan(attempt)
            intervals = [i for i in (plan.get("intervals") or []) if isinstance(i, dict)]
            if not intervals:
                self.record_skip("repair_run", "plan is empty")
                self.record_skip("repair_merge", "plan is empty")
                return
            if plan.get("within_budget") is False:
                self.record_skip("repair_run", "plan outside budget")
                self.record_skip("repair_merge", "plan outside budget")
                return
            self.stage_repair_run(attempt)
            self.stage_repair_merge(attempt)
            self.stage_gate_after(attempt)
            base_doc = read_json(self.segments if attempt == 1 else self.merged_path(attempt - 1))
            merged_doc = read_json(self.merged_path(attempt))
            gained = union_length(merged_doc.get("segments")) - union_length(base_doc.get("segments"))
            if gained < 0.5:
                note(f"[repair] прирост покрытия {gained:.1f} с — повторного ремонта не будет")
                return
            verdict = str(read_json(self.trust_after_path(attempt)).get("verdict") or "")
            if verdict == "ok" or uncovered_s(read_json(self.trust_after_path(attempt))) <= 0.5:
                return
            if attempt >= tries:
                return

    def stage_summary(self) -> str:
        trust = read_json(self.final_trust())
        verdict = str(trust.get("verdict") or "")
        route = self.route_doc()
        blocked = route_blocked(route)
        allowed = bool(route.get("summary_allowed", True))
        if blocked or verdict == "failed" or not allowed:
            reason = "route blocked" if blocked else ("verdict failed" if verdict == "failed"
                                                      else "summary_allowed=false")
            # a summary from an earlier attempt must not survive a forbidden verdict
            for name in ("summary.md", "summary.json", "summary.rejected.json"):
                stale = self.p("50_summary", name)
                if stale.exists():
                    stale.unlink()
                    note(f"[summary] удалён устаревший {name} ({reason})")
            self.record_skip("summary", reason)
            return "skipped"
        params = {"mode": route.get("summary_mode"), "verdict": verdict}
        inputs = {"trust": self.final_trust(), "transcript": self.effective()}
        summary_md = self.p("50_summary", "summary.md")
        summary_json = self.p("50_summary", "summary.json")
        rejected_json = self.p("50_summary", "summary.rejected.json")

        def refused() -> bool:
            candidate = summary_json if summary_json.is_file() else rejected_json
            if summary_md.exists() or not candidate.is_file():
                return False
            try:
                return str(read_json(candidate).get("status") or "") == "refused"
            except PipelineError:
                return False

        def action():
            argv = ["--trust", str(self.final_trust()), "--transcript", str(self.effective()),
                    "--segments", str(self.effective()), "--out", str(self.p("50_summary"))]
            return self.call_module("asr_summary.py", argv, "summary", allowed=(0, 2, 3, 4))

        status = self.run_stage("summary", inputs, params, [summary_md, summary_json], action,
                                outputs_check=lambda: summary_md.exists() or refused())
        card_path = summary_json if summary_json.is_file() else rejected_json
        if card_path.exists():
            card = read_json(card_path)
            self.summary_refused = str(card.get("status") or "") == "refused"
            if self.summary_refused:
                if card_path != rejected_json:
                    os.replace(card_path, rejected_json)
                stage = next(s for s in self.state.data["stages"] if s["name"] == "summary")
                stage["outputs"] = [relpath(self.out, rejected_json), "50_summary/summary.rejected.md"]
                self.state.save()
                note("[summary] guard отклонил саммари — нужен человек")
        return status

    def stage_verify(self) -> None:
        report_path = self.p("60_verify", "report.json")
        params = {"schema": RUN_SCHEMA}
        inputs = {"manifest": self.manifest, "trust": self.final_trust(), "segments": self.effective(),
                  "route": self.route}
        digest = inputs_hash("verify", params, inputs)
        fingerprint = {"params": params, "inputs": {k: relpath(self.out, v) for k, v in inputs.items()}}
        self.state.record("verify", "pending", digest, [relpath(self.out, report_path)], EXIT_OK)
        self.state.save()
        report = verify_run(self.out, _in_progress=True)
        write_json(report_path, report)
        self.state.record("verify", "done" if report["ok"] else "failed", digest,
                          [relpath(self.out, report_path)],
                          EXIT_OK if report["ok"] else EXIT_FAILED,
                          extra={"violations": report["violations"], "checked": report["checked"],
                                 "fingerprint": fingerprint})
        self.state.data["verify"] = {"ok": report["ok"], "violations": report["violations"],
                                     "checked": report["checked"]}
        self.state.save()
        if not report["ok"]:
            note("[verify] нарушения: " + ", ".join(v["code"] for v in report["violations"]))

    # -- outcome -----------------------------------------------------------
    def _set_outcome(self) -> None:
        trust = read_json(self.final_trust())
        verdict = str(trust.get("verdict") or "failed")
        if verdict not in VERDICT_EXIT:
            verdict = "failed"
        route = self.route_doc()
        if route_blocked(route):
            verdict = "blocked"
        exit_code = VERDICT_EXIT[verdict]
        if not bool((self.state.data.get("verify") or {}).get("ok", True)):
            verdict, exit_code = "failed", EXIT_FAILED
        if self.summary_refused and exit_code in (EXIT_OK, EXIT_PARTIAL):
            verdict, exit_code = "blocked", EXIT_BLOCKED  # guard refusal requires a human
        manifest = read_json(self.manifest)
        self.state.data.update({
            "verdict": verdict,
            "exit_code": exit_code,
            "unresolved_s": uncovered_s(trust),
            "effective_transcript": relpath(self.out, self.effective()),
            "final_trust": relpath(self.out, self.final_trust()),
            "summary_refused": bool(self.summary_refused),
            "audio": {"path": manifest["audio"]["path"], "sha256": manifest["audio"]["sha256"],
                      "duration_s": manifest["audio"]["duration_s"]},
            "engines": self.engines_record(),
            "finished_at": now_iso(),
        })
        self.state.save()
        self.exit_code = exit_code

    def collapse(self, code: int) -> int:
        if self.legacy_exit and code in (EXIT_FAILED, EXIT_BLOCKED):
            return EXIT_PARTIAL
        return code

    def payload(self) -> dict:
        data = self.state.data
        return {"schema": REPORT_SCHEMA, "run_dir": str(self.out), "verdict": data.get("verdict"),
                "exit_code": data.get("exit_code"), "unresolved_s": data.get("unresolved_s"),
                "stages": [{"name": s.get("name"), "status": s.get("status")}
                           for s in data.get("stages") or []],
                "effective_transcript": data.get("effective_transcript"),
                "attempts": data.get("attempts"),
                "summary_refused": bool(data.get("summary_refused")),
                "verify_ok": bool((data.get("verify") or {}).get("ok", True))}

    # -- run ---------------------------------------------------------------
    def run(self) -> int:
        if self.audio is None:
            raise PipelineError("E_ARGS", "--audio обязателен", EXIT_ARGS)
        if not self.audio.is_file():
            raise PipelineError("E_AUDIO", f"нет аудиофайла {self.audio}", EXIT_ARGS)
        for name, path in self.peers.items():
            if not path.is_file():
                raise PipelineError("E_PEER", f"нет peer-транскрипта {name}: {path}", EXIT_ARGS)
        self.out.mkdir(parents=True, exist_ok=True)
        if not self.resume:
            if self.state.prev:
                self.reset_outputs()
            self.state.data["attempts"] = {"prep": 0, "repair": 0}
        (self.out / "logs").mkdir(exist_ok=True)

        self.stage_ingest()
        if self.stage_asr() == "done":
            # A model/command can change while emitting identical text. Never
            # reuse repairs (or stale final_trust pointers) from that decoder.
            self._invalidate_downstream()
            self.state.prev.pop("ensemble", None)
        if self.ensemble_names and self.stage_ensemble() == "done":
            self._invalidate_downstream()
        self.stage_gate()
        self.stage_route()
        route = self.route_doc()
        prep = [s for s in route.get("steps") or [] if isinstance(s, dict) and s.get("kind") != "repair"]
        if prep:
            self.state.data["deferred"] = [str(s.get("id") or s.get("kind")) for s in prep]
            self.state.data.setdefault("notes", []).append(
                "route prep steps recognized but not executed by the orchestrator: "
                + ", ".join(str(s.get("id") or s.get("kind")) for s in prep))

        if route_blocked(route):
            self.record_skip("repair", "route blocked")
            self.stage_summary()
        else:
            self.stage_repair()
            if not self._regate_done:
                attempt = int((self.state.data.get("attempts") or {}).get("repair") or 0)
                if attempt and self.merged_path(attempt).exists():
                    self.stage_gate_after(attempt)
                else:
                    self.record_skip("gate_after", "no repaired transcript")
            self.stage_summary()

        self._set_outcome()
        self.stage_verify()
        self._set_outcome()
        print(json.dumps(self.payload(), ensure_ascii=False, indent=1))
        return self.collapse(self.exit_code)


# -------------------------------------------------------------------------- verify

def verify_run(run_dir, *, _in_progress: bool = False) -> dict:
    """Recompute the run-level invariants from the artifacts on disk."""
    run_dir = Path(run_dir).expanduser().resolve()
    violations: list[dict] = []
    checked = 0

    def bad(code: str, detail: str) -> None:
        violations.append({"code": code, "detail": detail})

    def report() -> dict:
        return {"schema": VERIFY_SCHEMA, "run_dir": str(run_dir), "ok": not violations,
                "violations": violations, "checked": checked}

    run_path = run_dir / "run.json"
    if not run_path.exists():
        bad("E_NO_RUN", f"нет {run_path}")
        return report()
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        bad("E_RUN_JSON", f"{run_path}: {exc}")
        return report()
    checked += 1
    if not isinstance(run, dict):
        bad("E_RUN_JSON", "run.json — не объект")
        return report()
    if run.get("schema") != RUN_SCHEMA:
        bad("E_RUN_SCHEMA", f"schema={run.get('schema')!r}, ожидался {RUN_SCHEMA!r}")

    stages = [s for s in run.get("stages") or [] if isinstance(s, dict)]
    names = [str(s.get("name")) for s in stages]
    checked += 1
    for required in ("ingest", "asr", "gate", "route", "summary", "verify"):
        if required not in names:
            bad("E_STAGE_MISSING", f"в run.json нет стадии {required}")
    if len(names) != len(set(names)):
        bad("E_STAGE_INCOMPLETE", "повторяющиеся стадии в run.json")
    minimum_outputs = {"ingest": ["00_input/manifest.json", "00_input/engines.json"],
                       "asr": ["10_asr/segments.json"], "gate": ["20_gate/trust.json"],
                       "route": ["30_route/route.json"], "repair_plan": ["40_repair/plan.json"],
                       "ensemble": ["10_asr/segments.json", "10_asr/ensemble/report.json"],
                       "repair_run": ["40_repair/additions.json"], "verify": ["60_verify/report.json"]}
    allowed_stage_exits = {"ingest": (0,), "asr": (0,), "ensemble": (0,), "repair": (0,),
                           "gate": (0, 2), "gate_after": (0, 2),
                           "repair_plan": (0, 2), "repair_run": (0, 2), "repair_merge": (0, 2),
                           "route": (0, 2, 3, 4), "summary": (0, 2, 3, 4), "verify": (0, 3)}
    for stage in stages:
        checked += 1
        name = stage.get("name")
        exit_code = stage.get("exit")
        if type(exit_code) is not int or exit_code not in allowed_stage_exits.get(name, ()):
            bad("E_STAGE_EXIT", f"{name}: exit={exit_code!r}, допустимы {allowed_stage_exits.get(name, ())}")
        pending = _in_progress and name == "verify" and stage.get("status") == "pending"
        if pending:
            continue  # private in-process check; public verify never accepts pending
        if stage.get("status") not in ("done", "skipped"):
            bad("E_STAGE_INCOMPLETE", f"{name}: status={stage.get('status')!r}")
        spec = stage.get("fingerprint")
        if not stage.get("inputs_hash") or not isinstance(spec, dict):
            bad("E_STAGE_INCOMPLETE", f"{name}: нет inputs_hash/fingerprint")
        else:
            try:
                resolved = {k: run_dir / v if v is not None else None for k, v in spec["inputs"].items()}
                if any(v is not None and not v.is_file() for v in resolved.values()):
                    bad("E_STAGE_INCOMPLETE", f"{name}: отсутствует входной файл")
                digest = inputs_hash(name, spec["params"], resolved)
                if digest != stage["inputs_hash"]:
                    bad("E_STAGE_INCOMPLETE", f"{name}: inputs_hash изменился")
            except (KeyError, TypeError, AttributeError, OSError) as exc:
                bad("E_STAGE_INCOMPLETE", f"{name}: неверный fingerprint: {exc}")
        outputs = stage.get("outputs")
        no_op = (stage.get("status") == "skipped" and isinstance(spec, dict)
                 and bool((spec.get("params") or {}).get("skip_reason"))
                 and name in {"repair", "repair_run", "repair_merge", "gate_after", "summary", "ensemble"})
        if not isinstance(outputs, list) or (not outputs and not no_op):
            bad("E_STAGE_INCOMPLETE", f"{name}: outputs пусты/не список")
            outputs = []
        if not no_op:
            expected_outputs = minimum_outputs.get(name, [])
            if name == "asr" and "ensemble" in names:
                engine = ((spec or {}).get("params") or {}).get("engine")
                expected_outputs = ["10_asr/primary_attempt.json"]
                attempt_path = run_dir / "10_asr/primary_attempt.json"
                try:
                    if read_json(attempt_path).get("ok"):
                        expected_outputs.append(f"10_asr/{engine}.json")
                except (PipelineError, AttributeError):
                    bad("E_STAGE_INCOMPLETE", "asr: нет результата основной попытки")
            for rel in expected_outputs:
                if rel not in outputs:
                    bad("E_STAGE_INCOMPLETE", f"{name}: не объявлен output {rel}")
        for rel in outputs:
            if not isinstance(rel, str) or not (run_dir / rel).is_file():
                bad("E_STAGE_INCOMPLETE", f"{name}: отсутствует output {rel!r}")
    ranked = [STAGE_RANK[name] for name in names if name in STAGE_RANK]
    if ranked != sorted(ranked):
        bad("E_STAGE_ORDER", f"порядок стадий нарушен: {names}")

    required_artifacts = {
        "00_input/manifest.json": "E_MISSING_MANIFEST",
        "10_asr/segments.json": "E_MISSING_SEGMENTS",
        "20_gate/trust.json": "E_MISSING_TRUST",
        "30_route/route.json": "E_MISSING_ROUTE",
    }
    for rel, code in required_artifacts.items():
        checked += 1
        if not (run_dir / rel).exists():
            bad(code, f"нет артефакта {rel}")

    manifest = {}
    manifest_path = run_dir / "00_input" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            bad("E_MANIFEST_JSON", f"{manifest_path}: {exc}")
        else:
            if not isinstance(manifest, dict):
                bad("E_MANIFEST_JSON", f"{manifest_path}: верхний уровень не объект")
                manifest = {}
            audio = manifest.get("audio") or {}
            checked += 1
            if not audio.get("sha256") or not audio.get("duration_s"):
                bad("E_MANIFEST_FIELDS", "в manifest.json нет audio.sha256/audio.duration_s")
    duration = (manifest.get("audio") or {}).get("duration_s")

    def check_segments(path: Path, label: str) -> None:
        nonlocal checked
        checked += 1
        if not path.exists():
            bad("E_MISSING_SEGMENTS", f"нет {label}: {path}")
            return
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            bad("E_SEGMENTS_JSON", f"{label}: {exc}")
            return
        if not isinstance(document, dict):
            bad("E_SEGMENTS_JSON", f"{label}: верхний уровень не объект")
            return
        if str(document.get("schema")) != SEGMENTS_SCHEMA:
            bad("E_SEGMENTS_SCHEMA", f"{label}: schema={document.get('schema')!r}")
        rows = document.get("segments") or []
        if not isinstance(rows, list):
            bad("E_SEGMENTS_FIELD", f"{label}: segments — не список")
            rows = []
        if not rows:
            bad("E_SEGMENTS_EMPTY", f"{label}: пустой список сегментов")
        last_start = None
        seen_ids = set()
        for index, row in enumerate(rows):
            checked += 1
            if not isinstance(row, dict):
                bad("E_SEGMENTS_FIELD", f"{label}[{index}]: не объект")
                continue
            try:
                start, end = float(row["start"]), float(row["end"])
            except (KeyError, TypeError, ValueError):
                bad("E_SEGMENTS_FIELD", f"{label}[{index}]: нет числовых start/end")
                continue
            if not start < end:
                bad("E_SEGMENTS_BOUNDS", f"{label}[{index}]: start={start} >= end={end}")
            if last_start is not None and start < last_start - 1e-6:
                bad("E_SEGMENTS_ORDER", f"{label}[{index}]: start={start} после {last_start}")
            last_start = start
            if not str(row.get("text") or "").strip():
                bad("E_SEGMENTS_TEXT", f"{label}[{index}]: пустой text")
            if duration is not None and end > float(duration) + 0.1:
                bad("E_SEGMENTS_DURATION", f"{label}[{index}]: end={end} > duration+0.1")
            identifier = row.get("id")
            if identifier in seen_ids:
                bad("E_SEGMENTS_ID", f"{label}[{index}]: повтор id={identifier!r}")
            seen_ids.add(identifier)
        audio = document.get("audio") or {}
        if not isinstance(audio, dict):
            bad("E_SEGMENTS_AUDIO", f"{label}: audio — не объект")
        elif (manifest.get("audio") or {}).get("sha256") and audio.get("sha256") != manifest["audio"]["sha256"]:
            bad("E_SEGMENTS_AUDIO", f"{label}: audio.sha256 не совпадает с manifest.json")

    check_segments(run_dir / "10_asr" / "segments.json", "10_asr/segments.json")
    effective_rel = run.get("effective_transcript") or "10_asr/segments.json"
    if effective_rel != "10_asr/segments.json":
        check_segments(run_dir / effective_rel, effective_rel)

    recorded_trust = run.get("final_trust")
    if not isinstance(recorded_trust, str) or not (run_dir / recorded_trust).is_file():
        bad("E_STAGE_INCOMPLETE", f"final_trust ссылается на отсутствующий файл: {recorded_trust!r}")
        recorded_trust = None
    trust_path = (run_dir / recorded_trust) if recorded_trust else (run_dir / "21_gate_after" / "trust.json")
    if not trust_path.exists():
        trust_path = run_dir / "21_gate_after" / "trust.json"
    if not trust_path.exists():
        trust_path = run_dir / "20_gate" / "trust.json"
    trust: dict = {}
    checked += 1
    if not trust_path.exists():
        bad("E_MISSING_TRUST", "нет ни 21_gate_after/trust.json, ни 20_gate/trust.json")
    else:
        try:
            trust = json.loads(trust_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            bad("E_TRUST_JSON", f"{trust_path}: {exc}")
            trust = {}
        if not isinstance(trust, dict):
            bad("E_TRUST_JSON", f"{trust_path}: верхний уровень не объект")
            trust = {}
    if trust:
        if str(trust.get("schema")) != TRUST_SCHEMA:
            bad("E_TRUST_SCHEMA", f"schema={trust.get('schema')!r}, ожидался {TRUST_SCHEMA!r}")
        verdict = str(trust.get("verdict") or "")
        if verdict not in ("ok", "partial", "failed"):
            bad("E_TRUST_VERDICT", f"verdict={verdict!r}")
        block_reason = False
        for reason in trust.get("reasons") or []:
            checked += 1
            if not isinstance(reason, dict):
                bad("E_TRUST_REASON", "reason не объект")
                continue
            if reason.get("severity") not in ("warn", "block"):
                bad("E_TRUST_SEVERITY", f"{reason.get('code')}: severity={reason.get('severity')!r}")
            if reason.get("severity") == "block":
                block_reason = True
        reason_verdict = "failed" if block_reason else ("partial" if trust.get("reasons") else "ok")
        if verdict != reason_verdict:
            bad("E_TRUST_VERDICT", f"по причинам ожидался {reason_verdict!r}, verdict={verdict!r}")
            bad("E_EXIT_MISMATCH", "final_trust.verdict не соответствует severity причин")
        checked += 1
        if not isinstance(trust.get("uncovered_all"), list):
            bad("E_TRUST_UNCOVERED", "в trust.json нет uncovered_all")
        else:
            total = uncovered_s(trust)
            declared = run.get("unresolved_s")
            if declared is None:
                bad("E_RUN_UNRESOLVED", "в run.json нет unresolved_s")
            elif abs(float(declared) - total) > 0.11:
                bad("E_UNRESOLVED_MISMATCH",
                    f"run.json unresolved_s={declared}, uncovered_all={total}")
        union = trust.get("coverage_union")
        if isinstance(union, dict) and union.get("covered_s") is not None and union.get("active_s") is not None:
            checked += 1
            try:
                if float(union["covered_s"]) > float(union["active_s"]) + 0.1:
                    bad("E_COVERAGE", f"covered_s={union['covered_s']} > active_s={union['active_s']}")
            except (TypeError, ValueError):
                bad("E_COVERAGE_MISMATCH", "coverage_union содержит нечисловые секунды")

    # Recompute from audio, not from declared activity/coverage (no ML providers).
    # The initial gate belongs to 10_asr; the final gate belongs to the repaired
    # transcript, when present. A correct final gate cannot hide a forged first one.
    try:
        import math
        import numpy as np
        import soundfile as sf
        from asr_gate import coverage, energy_activity

        audio_path = Path((manifest.get("audio") or {})["path"])
        samples, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1).astype(np.float32)
        energy = energy_activity(samples, sr)

        def same_number(actual, expected) -> bool:
            if expected is None:
                return actual is None
            return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                    and math.isfinite(actual) and abs(actual - expected) <= 0.01 + 1e-9)

        def same_intervals(actual, expected) -> bool:
            return (isinstance(actual, list) and len(actual) == len(expected)
                    and all(isinstance(a, dict)
                            and all(same_number(a.get(k), b[k]) for k in ("start", "end"))
                            for a, b in zip(actual, expected)))

        pairs = [(run_dir / "20_gate/trust.json", run_dir / "10_asr/segments.json")]
        if trust_path != pairs[0][0]:
            pairs.append((trust_path, run_dir / effective_rel))
        for gate_path, transcript in pairs:
            if not gate_path.is_file() or not transcript.is_file():
                continue
            gate_doc = read_json(gate_path)
            reasons = gate_doc.get("reasons") or []
            severities = {r.get("severity") for r in reasons if isinstance(r, dict)}
            expected_verdict = "failed" if "block" in severities else ("partial" if reasons else "ok")
            if gate_doc.get("verdict") != expected_verdict:
                bad("E_EXIT_MISMATCH", f"{gate_path}: verdict не соответствует severity причин")
            rows = read_json(transcript).get("segments") or []
            expected = coverage(rows, len(samples) / sr, energy)
            union_doc = gate_doc.get("coverage_union") or {}
            primary_doc = (gate_doc.get("coverage") or {}).get(gate_doc.get("primary")) or {}
            checked += 1
            for key, source_key in (("covered_s", "covered_s"), ("active_s", "union_active_s"),
                                    ("coverage_union", "coverage_union")):
                if not same_number(union_doc.get(key), expected[source_key]):
                    bad("E_COVERAGE_MISMATCH", f"{gate_path}: coverage_union.{key}="
                        f"{union_doc.get(key)!r}, пересчитано {expected[source_key]!r}")
            for key in ("covered_s", "union_active_s", "coverage", "coverage_union", "tail_uncovered_s"):
                if not same_number(primary_doc.get(key), expected[key]):
                    bad("E_COVERAGE_MISMATCH", f"{gate_path}: primary.{key} не совпадает с аудио/сегментами")
            if not same_number((gate_doc.get("activity") or {}).get("energy_active_s"), energy["active_s"]):
                bad("E_COVERAGE_MISMATCH", f"{gate_path}: energy_active_s не совпадает с аудио")
            if not same_intervals(gate_doc.get("uncovered_all"), expected["speech_gaps"]):
                bad("E_COVERAGE_MISMATCH", f"{gate_path}: uncovered_all не совпадает с union-дырами")
            if not same_intervals(primary_doc.get("speech_gaps"), expected["speech_gaps"]):
                bad("E_COVERAGE_MISMATCH", f"{gate_path}: primary.speech_gaps не совпадает с union-дырами")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, PipelineError) as exc:
        bad("E_COVERAGE_MISMATCH", f"не удалось пересчитать покрытие: {exc}")

    route: dict = {}
    route_path = run_dir / "30_route" / "route.json"
    if route_path.exists():
        try:
            route = json.loads(route_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            bad("E_ROUTE_JSON", f"{route_path}: {exc}")
        else:
            if not isinstance(route, dict):
                bad("E_ROUTE_JSON", f"{route_path}: верхний уровень не объект")
                route = {}
            else:
                checked += 1
                if str(route.get("schema")) != "asr.route/1":
                    bad("E_ROUTE_SCHEMA", f"schema={route.get('schema')!r}")
                if not isinstance(route.get("steps"), list):
                    bad("E_ROUTE_STEPS", "steps — не список")

    attempts = (run.get("attempts") or {}).get("repair") or 0
    required_repair = ("repair_plan", "repair_run", "repair_merge", "gate_after") if attempts else ()
    if any(s.get("kind") == "repair" for s in route.get("steps") or [] if isinstance(s, dict)) and not route_blocked(route):
        required_repair = tuple(dict.fromkeys(("repair_plan", *required_repair)))
    for name in required_repair:
        if name not in names:
            bad("E_STAGE_INCOMPLETE", f"нет обязательной стадии ремонта {name}")
    summary_dir = run_dir / "50_summary"
    summary_md = summary_dir / "summary.md"
    summary_json_path = summary_dir / "summary.json"
    rejected_json_path = summary_dir / "summary.rejected.json"
    card_path = summary_json_path if summary_json_path.is_file() else rejected_json_path
    card: dict = {}
    if card_path.exists():
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            bad("E_SUMMARY_JSON", f"{summary_json_path}: {exc}")
            card = {}
    if not isinstance(card, dict):
        bad("E_SUMMARY_MISMATCH", "summary.json: верхний уровень не объект")
        card = {}
    refused = str(card.get("status") or "") == "refused"
    verdict = str(trust.get("verdict") or "")
    blocked = route_blocked(route)
    allowed = bool(route.get("summary_allowed", True)) if route else True
    forbidden = (blocked or verdict == "failed" or not allowed
                 or (run.get("verdict") in {"failed", "blocked"} and not refused))
    checked += 1
    if forbidden and (summary_md.exists() or summary_json_path.exists()):
        bad("E_SUMMARY_FORBIDDEN",
            f"summary.md есть при verdict={verdict!r}, blocked={blocked}, summary_allowed={allowed}")
    if not forbidden and not summary_md.exists() and not refused:
        bad("E_SUMMARY_MISSING", "нет 50_summary/summary.md при не-failed вердикте")
    if refused:
        checked += 1
        if not (summary_dir / "summary.rejected.md").exists():
            bad("E_SUMMARY_REJECTED", "guard отказал от саммари, но summary.rejected.md отсутствует")
        if run.get("summary_refused") is not True:
            bad("E_SUMMARY_REFUSED_FLAG", "summary.json refused, но run.json этого не отмечает")
    if summary_md.exists():
        checked += 1
        text = summary_md.read_text(encoding="utf-8", errors="replace")
        if verdict == "partial" and "⚠" not in text:
            bad("E_SUMMARY_CAVEAT", "при partial в summary.md нет строки ⚠ с кодами")
    if card:
        checked += 1
        if card.get("verdict") not in (None, verdict):
            bad("E_SUMMARY_VERDICT", f"summary.json verdict={card.get('verdict')!r}, trust={verdict!r}")
        if card.get("trusted") is True and verdict != "ok":
            bad("E_SUMMARY_TRUSTED", f"summary.json trusted=true при verdict={verdict!r}")
        guard = card.get("guard")
        # Rejected diagnostics live under summary.rejected.json; any published
        # summary.json must pass every boolean guard, without a refused bypass.
        if summary_json_path.exists():
            flags = {"numbers_ok", "tasks_quoted", "caveat_present"}
            if isinstance(guard, dict):
                flags |= set(guard) - {"numbers_checked", "violations"}
            if not isinstance(guard, dict) or any(guard.get(k) is not True for k in flags):
                bad("E_SUMMARY_MISMATCH", "summary.json: guard-проверки должны быть true")

    if summary_json_path.exists() and not card:
        bad("E_SUMMARY_MISMATCH", "summary.json: пустой/неверный объект, нет guard")
    if refused and (summary_md.exists() or summary_json_path.exists()):
        bad("E_SUMMARY_FORBIDDEN", "summary.md/summary.json опубликован вопреки отказу guard")
    expected_verdict = "blocked" if blocked or refused else verdict
    expected_exit = VERDICT_EXIT.get(expected_verdict)
    if run.get("verdict") != expected_verdict or run.get("exit_code") != expected_exit:
        bad("E_EXIT_MISMATCH", f"run verdict/exit={run.get('verdict')!r}/{run.get('exit_code')!r}, "
            f"ожидалось {expected_verdict!r}/{expected_exit!r}")
    if card and not refused and card.get("verdict") != verdict:
        bad("E_SUMMARY_MISMATCH", "summary.json verdict не совпадает с final_trust")

    if summary_md.exists():
        import math
        from asr_summary import NUM_SPAN, quality_lines

        text = summary_md.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?ms)^## Качество записи\s*$(.*?)(?=^## |\Z)", text)
        quality = match.group(1) if match else ""
        lines = [line.strip() for line in quality.splitlines() if line.strip()]
        number = r"([-+−]?(?:\d+(?:[.,]\d+)?)(?:[eE][-+]?\d+)?)"
        matched_lines = set()
        # Derive every metric and its position from the publisher's template,
        # rather than maintaining a second, inevitably incomplete metric list.
        for template in quality_lines(trust):
            tokens = list(NUM_SPAN.finditer(template))
            if not tokens:
                continue
            parts, expected, cursor = [], [], 0
            for token in tokens:
                parts.extend((re.escape(template[cursor:token.start()]), number))
                expected.append(float(token.group().replace(",", ".")))
                cursor = token.end()
            parts.append(re.escape(template[cursor:]))
            pattern = "".join(parts)
            matches = [(i, re.fullmatch(pattern, line, flags=re.IGNORECASE))
                       for i, line in enumerate(lines)]
            matches = [(i, m) for i, m in matches if m is not None]
            checked += len(expected)
            matched_lines.update(i for i, _ in matches)
            if len(matches) != 1:
                bad("E_SUMMARY_MISMATCH", f"summary.md: ожидалась одна строка метрик {template!r}")
                continue
            values = [float(v.replace(",", ".").replace("−", "-")) for v in matches[0][1].groups()]
            if any(not math.isfinite(value) or abs(value - target) > 0.01 + 1e-9
                   for value, target in zip(values, expected)):
                bad("E_SUMMARY_MISMATCH", f"summary.md {template!r}: {values!r}, trust={expected!r}")
        for i, line in enumerate(lines):
            if i not in matched_lines and NUM_SPAN.search(line):
                bad("E_SUMMARY_MISMATCH", f"summary.md: лишние/неверные числовые метрики {line!r}")

    return report()


# ----------------------------------------------------------------------------- CLI

class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse would exit 2; the contract wants 64
        print(json.dumps({"error": {"code": "E_ARGS", "detail": message}, "exit_code": EXIT_ARGS},
                         ensure_ascii=False))
        raise SystemExit(EXIT_ARGS)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Оркестратор пайплайна ASR (состояние asr.run/1)")
    sub = parser.add_subparsers(dest="cmd")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stages-dir", help="каталог модулей стадий (по умолчанию рядом с asr_pipeline.py)")

    run = sub.add_parser("run", parents=[common], help="полный прогон")
    run.add_argument("--audio", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--engines", help="engines.json (по умолчанию рядом с asr_pipeline.py)")
    run.add_argument("--engine", help="имя движка (по умолчанию первый/primary из engines.json)")
    run.add_argument("--ensemble", metavar="NAME[,NAME...]|all",
                     help="добавить ensemble:true движки и использовать medoid-ансамбль как primary")
    run.add_argument("--peer", action="append", default=[], metavar="NAME=PATH",
                     help="дополнительный транскрипт для gate (можно несколько)")
    run.add_argument("--language", default="ru")
    run.add_argument("--vad", default="silero", choices=["silero", "energy", "off"])
    run.add_argument("--lid", default="whisper", choices=["whisper", "declared", "off"])
    run.add_argument("--budget-frac", type=float, default=DEFAULT_BUDGET_FRAC)
    run.add_argument("--resume", action="store_true", help="продолжить прогон, пропуская неизменённые стадии")
    run.add_argument("--legacy-exit", action="store_true", help="схлопнуть коды 3 и 4 в 2")

    gate = sub.add_parser("gate", parents=[common], help="только gate по готовому транскрипту")
    gate.add_argument("--audio", required=True)
    gate.add_argument("--segments", required=True)
    gate.add_argument("--peer", action="append", default=[], metavar="NAME=PATH")
    gate.add_argument("--primary", default="whisper")
    gate.add_argument("--language", default="ru")
    gate.add_argument("--vad", default="silero", choices=["silero", "energy", "off"])
    gate.add_argument("--lid", default="whisper", choices=["whisper", "declared", "off"])
    gate.add_argument("--out", required=True)

    route = sub.add_parser("route", parents=[common], help="только route по trust.json")
    route.add_argument("--trust", required=True)
    route.add_argument("--out", required=True)
    route.add_argument("--budget-frac", type=float, default=DEFAULT_BUDGET_FRAC)

    repair = sub.add_parser("repair", parents=[common], help="plan | run | merge из asr_repair.py")
    repair.add_argument("--mode", required=True, choices=["plan", "run", "merge"])
    repair.add_argument("--trust")
    repair.add_argument("--route")
    repair.add_argument("--plan")
    repair.add_argument("--audio")
    repair.add_argument("--engines")
    repair.add_argument("--base")
    repair.add_argument("--additions")
    repair.add_argument("--out", required=True)

    summary = sub.add_parser("summarize", parents=[common], help="только summary по trust.json")
    summary.add_argument("--trust", required=True)
    summary.add_argument("--transcript", required=True)
    summary.add_argument("--segments")
    summary.add_argument("--out", required=True)

    verify = sub.add_parser("verify", parents=[common], help="пересчитать инварианты прогона")
    verify.add_argument("--run", required=True)

    status = sub.add_parser("status", parents=[common], help="состояние прогона")
    status.add_argument("--run", required=True)
    status.add_argument("--legacy-exit", action="store_true")
    return parser


def next_action(run: dict) -> str:
    verdict = run.get("verdict")
    if run.get("summary_refused"):
        return "саммари отклонено guard — нужен человек"
    if verdict == "ok":
        return "готово: текст можно отдавать"
    if verdict == "partial":
        return "проверить summary и спорные места (unresolved_s)"
    if verdict == "failed":
        return "вердикт failed: сменить движок/параметры и повторить run"
    if verdict == "blocked":
        return "нужен человек: маршрут заблокирован"
    return "прогон не завершён — запустить run --resume"


def cmd_status(args) -> int:
    run_dir = Path(args.run).expanduser().resolve()
    path = run_dir / "run.json"
    if not path.exists():
        raise PipelineError("E_NO_RUN", f"нет {path}", EXIT_FAILED)
    run = read_json(path)
    payload = {"schema": "asr.status/1", "run_dir": str(run_dir), "verdict": run.get("verdict"),
               "exit_code": run.get("exit_code"), "unresolved_s": run.get("unresolved_s"),
               "attempts": run.get("attempts"), "verify": run.get("verify"),
               "stages": [{"name": s.get("name"), "status": s.get("status")}
                          for s in run.get("stages") or []],
               "next": next_action(run)}
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    code = int(run.get("exit_code") if run.get("exit_code") is not None else EXIT_FAILED)
    if args.legacy_exit and code in (EXIT_FAILED, EXIT_BLOCKED):
        return EXIT_PARTIAL
    return code


def cmd_verify(args) -> int:
    run_dir = Path(args.run).expanduser().resolve()
    report = verify_run(run_dir)
    if (run_dir / "run.json").exists():
        write_json(run_dir / "60_verify" / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return EXIT_OK if report["ok"] else EXIT_FAILED


def cmd_passthrough(args) -> int:
    stages_dir = Path(args.stages_dir).expanduser().resolve() if args.stages_dir else HERE
    if args.cmd == "gate":
        argv = ["--audio", str(Path(args.audio).expanduser().resolve()),
                "--transcript", f"{args.primary}={Path(args.segments).expanduser().resolve()}"]
        for spec in args.peer:
            name, sep, path = str(spec).partition("=")
            if not sep or not name or not path:
                raise PipelineError("E_ARGS", f"--peer ожидает NAME=PATH, получено {spec!r}", EXIT_ARGS)
            argv += ["--transcript", f"{name}={Path(path).expanduser().resolve()}"]
        argv += ["--primary", args.primary, "--vad", args.vad, "--lid", args.lid,
                 "--language", args.language, "--out", str(Path(args.out).expanduser().resolve())]
        return run_module(stages_dir, "asr_gate.py", argv, label="gate", allowed=(0, 2)).returncode
    if args.cmd == "route":
        argv = ["--trust", str(Path(args.trust).expanduser().resolve()),
                "--out", str(Path(args.out).expanduser().resolve()),
                "--budget-frac", str(args.budget_frac)]
        return run_module(stages_dir, "asr_router.py", argv, label="route", allowed=(0, 2)).returncode
    if args.cmd == "summarize":
        argv = ["--trust", str(Path(args.trust).expanduser().resolve()),
                "--transcript", str(Path(args.transcript).expanduser().resolve())]
        if args.segments:
            argv += ["--segments", str(Path(args.segments).expanduser().resolve())]
        argv += ["--out", str(Path(args.out).expanduser().resolve())]
        return run_module(stages_dir, "asr_summary.py", argv, label="summary", allowed=(0, 2)).returncode
    if args.cmd == "repair":
        need = {"plan": ("trust", "route"), "run": ("audio", "plan", "engines"),
                "merge": ("base", "additions")}[args.mode]
        missing = [key for key in need if not getattr(args, key)]
        if missing:
            raise PipelineError("E_ARGS", f"repair {args.mode}: не хватает {', '.join('--' + m for m in missing)}",
                                EXIT_ARGS)
        argv = [args.mode]
        for key in need:
            argv += [f"--{key}", str(Path(getattr(args, key)).expanduser().resolve())]
        argv += ["--out", str(Path(args.out).expanduser().resolve())]
        return run_module(stages_dir, "asr_repair.py", argv, label=f"repair_{args.mode}",
                          allowed=(0, 2, 3)).returncode
    raise PipelineError("E_ARGS", f"неизвестная подкоманда {args.cmd!r}", EXIT_ARGS)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.error("нужна подкоманда: run|gate|route|repair|summarize|verify|status")
    try:
        if args.cmd == "run":
            pipeline = Pipeline(args.audio, Path(args.out), engines=args.engines, primary=args.engine,
                                peer=args.peer, language=args.language, vad=args.vad, lid=args.lid,
                                budget_frac=args.budget_frac, resume=args.resume,
                                stages_dir=args.stages_dir, legacy_exit=args.legacy_exit, ensemble=args.ensemble)
            return pipeline.run()
        if args.cmd == "status":
            return cmd_status(args)
        if args.cmd == "verify":
            return cmd_verify(args)
        return cmd_passthrough(args)
    except PipelineError as exc:
        print(json.dumps({"error": {"code": exc.code, "detail": exc.detail},
                          "exit_code": exc.exit_code}, ensure_ascii=False))
        return exc.exit_code
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(json.dumps({"error": {"code": "E_INTERNAL", "detail": f"{type(exc).__name__}: {exc}"},
                          "exit_code": EXIT_ERROR}, ensure_ascii=False))
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
