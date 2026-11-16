#!/usr/bin/env python3
"""Convert legacy transcript formats into `asr.segments/1`.

Supported inputs (auto-detected):

1. flat list  `[{"start": 0.0, "end": 7.5, "text": "..."}, ...]`
2. nested     `{"segments": [...]}` (also `items` / `results` / `windows` / `chunks`)
3. x20 unit   `{"id": "u1", "unit": {"start": 1.0, "end": 3.0}, "status": "ok",
               "result": {"text": "..."}}` — timestamps come from the unit
               bounds (dict, pair, or "00:01:02.500-00:01:05.000" string) when
               the record itself carries none.

Anything that cannot be placed on the timeline is skipped and counted; a file
where *no* record has bounds raises `E_NO_TIMING` instead of silently returning
an empty transcript.

Usage:
  adapt_legacy.py --in legacy.json --out segments.json \
                  --audio rec.wav [--engine whisper] [--model large-v3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = "asr.segments/1"

# keys that may hold the record list, in priority order
LIST_KEYS = ("segments", "items", "results", "units", "windows", "chunks", "rows", "data")
TEXT_KEYS = ("text", "transcript", "normalized_text", "content", "sentence")
OK_STATUS = {"", "ok", "done", "success", "finished", "complete", "completed", "none", "null"}


class AdaptError(ValueError):
    """Conversion failure carrying a stable machine code in the message."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# --------------------------------------------------------------------------- time

_TS_SECONDS = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(?:s|sec|secs|с)?$")
_TS_COLON = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:[.,]\d+)?)$")
_TS_UNIT_SPLIT = re.compile(r"\s*(?:-|–|—|‒|−−|\.\.|…|→|—>)\s*")


def parse_ts(value) -> float:
    """Seconds from a number, `MM:SS(.ms)`, `HH:MM:SS(.ms)` or `12.5s`."""
    if isinstance(value, bool):
        raise AdaptError("E_TS", f"bool не является таймкодом: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        raise AdaptError("E_TS", "пустой таймкод")
    m = _TS_SECONDS.match(text)
    if m:
        return float(m.group(1))
    m = _TS_COLON.match(text)
    if m:
        hours = int(m.group(1) or 0)
        return hours * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    raise AdaptError("E_TS", f"не разобран таймкод: {value!r}")


def parse_unit(unit):
    """`(start, end)` from an x20 `unit` field, or None when it carries no bounds."""
    if unit is None:
        return None
    if isinstance(unit, dict):
        for lo_key, hi_key in (("start", "end"), ("begin", "end"), ("start", "stop"),
                               ("from", "to"), ("s", "e"), ("start_s", "end_s")):
            if unit.get(lo_key) is not None and unit.get(hi_key) is not None:
                return parse_ts(unit[lo_key]), parse_ts(unit[hi_key])
        for key in ("b", "bounds", "span", "range", "interval"):
            pair = unit.get(key)
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                return parse_ts(pair[0]), parse_ts(pair[1])
        if unit.get("start") is not None and unit.get("duration") is not None:
            start = parse_ts(unit["start"])
            return start, start + parse_ts(unit["duration"])
        return None
    if isinstance(unit, (list, tuple)) and len(unit) >= 2:
        return parse_ts(unit[0]), parse_ts(unit[1])
    if isinstance(unit, str):
        parts = [p for p in _TS_UNIT_SPLIT.split(unit.strip()) if p]
        if len(parts) == 2:
            return parse_ts(parts[0]), parse_ts(parts[1])
    return None


# ----------------------------------------------------------------------- structure

def _looks_like_record(obj: dict) -> bool:
    if any(k in obj for k in TEXT_KEYS):
        return True
    result = obj.get("result")
    return isinstance(result, (dict, str)) or "unit" in obj


def items_of(data) -> list:
    """The record list of a legacy payload (or the single record it wraps)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if data.get("schema") == SCHEMA and isinstance(data.get("segments"), list):
            return data["segments"]
        for key in LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return value
        if _looks_like_record(data):
            return [data]
    raise AdaptError("E_FORMAT", f"не найден список сегментов в {type(data).__name__}")


def detect_format(data) -> str:
    if isinstance(data, dict) and data.get("schema") == SCHEMA:
        return SCHEMA
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return "nested"
    records = items_of(data)
    if any(isinstance(r, dict) and ("unit" in r or "result" in r) for r in records):
        return "x20"
    return "flat"


def record_text(rec: dict) -> str:
    for key in TEXT_KEYS:
        value = rec.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    result = rec.get("result")
    if isinstance(result, str) and result.strip():
        return " ".join(result.split())
    if isinstance(result, dict):
        for key in TEXT_KEYS:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split())
        words = result.get("words")
        if isinstance(words, list):
            tokens = [str(w.get("word") or w.get("text") or "") for w in words if isinstance(w, dict)]
            joined = " ".join(t for t in tokens if t)
            if joined.strip():
                return " ".join(joined.split())
    words = rec.get("words")
    if isinstance(words, list):
        tokens = [str(w.get("word") or w.get("text") or "") for w in words if isinstance(w, dict)]
        joined = " ".join(t for t in tokens if t)
        if joined.strip():
            return " ".join(joined.split())
    return ""


def record_bounds(rec: dict, allow_unit: bool = True):
    """Explicit bounds first, x20 unit bounds second."""
    for lo_key, hi_key in (("start", "end"), ("start_s", "end_s"), ("t_start", "t_end"),
                           ("begin", "end"), ("from", "to"), ("start", "stop")):
        if rec.get(lo_key) is not None and rec.get(hi_key) is not None:
            return parse_ts(rec[lo_key]), parse_ts(rec[hi_key])
    for key in ("b", "bounds", "span", "range", "interval"):
        pair = rec.get(key)
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            return parse_ts(pair[0]), parse_ts(pair[1])
    for lo_key, dur_key in (("start", "duration"), ("start", "dur"), ("start_s", "duration")):
        if rec.get(lo_key) is not None and rec.get(dur_key) is not None:
            start = parse_ts(rec[lo_key])
            return start, start + parse_ts(rec[dur_key])
    if allow_unit and rec.get("unit") is not None:
        return parse_unit(rec["unit"])
    return None


def _conf(rec: dict):
    for key in ("conf", "confidence", "probability", "score"):
        value = rec.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), 4)
    return None


def _speaker(rec: dict):
    value = rec.get("speaker", rec.get("spk", rec.get("speaker_id")))
    if value is None or isinstance(value, bool):
        return None
    return str(value)


def _flags(rec: dict) -> list:
    out = []
    raw = rec.get("flags")
    if isinstance(raw, list):
        out += [str(f) for f in raw]
    status = rec.get("status")
    if isinstance(status, str) and status.strip().lower() not in OK_STATUS:
        out.append(f"legacy_status:{status.strip()}")
    return out


def _audio_block(audio, duration_hint=None) -> dict:
    audio = dict(audio or {})
    duration = audio.get("duration_s", duration_hint)
    return {
        "path": audio.get("path"),
        "sha256": audio.get("sha256"),
        "duration_s": round(float(duration), 2) if duration is not None else None,
        "sr": audio.get("sr"),
        "channels": audio.get("channels"),
    }


def _engine_block(engine=None, *, name="legacy", model=None, version=None,
                  language=None, level_sensitive=False) -> dict:
    if isinstance(engine, dict):
        out = {"name": engine.get("name") or name, "model": engine.get("model"),
               "version": engine.get("version"), "params": dict(engine.get("params") or {}),
               "level_sensitive": bool(engine.get("level_sensitive", level_sensitive))}
    else:
        out = {"name": name, "model": model, "version": version, "params": {},
               "level_sensitive": bool(level_sensitive)}
    if language:
        out["params"].setdefault("language", language)
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(data, *, audio=None, engine=None, meta=None, min_dur: float = 0.001) -> dict:
    """Return an `asr.segments/1` document for `data` (raises AdaptError)."""
    if isinstance(data, dict) and data.get("schema") == SCHEMA:
        fmt = SCHEMA
        records = data.get("segments") or []
        src_audio, src_engine, src_meta = data.get("audio"), data.get("engine"), data.get("meta")
    else:
        fmt = detect_format(data)
        records = items_of(data)
        if isinstance(data, dict):
            src_audio, src_engine, src_meta = data.get("audio"), data.get("engine"), data.get("meta")
        else:
            src_audio = src_engine = src_meta = None

    rows: list[dict] = []
    skipped_bad = skipped_untimed = clamped = 0
    duration = (audio or {}).get("duration_s")
    if duration is None:
        duration = (src_audio or {}).get("duration_s")

    for index, rec in enumerate(records):
        if not isinstance(rec, dict):
            skipped_bad += 1
            continue
        text = record_text(rec)
        bounds = record_bounds(rec)
        if bounds is None:
            skipped_untimed += 1
            continue
        if not text:
            skipped_bad += 1
            continue
        start, end = bounds
        if end < start:
            start, end = end, start
        if end - start < min_dur:
            skipped_bad += 1
            continue
        if duration is not None and end > float(duration) + 0.1:
            end = float(duration)
            if end - start < min_dur:
                skipped_bad += 1
                continue
            clamped += 1
        if start < 0:
            start = 0.0
            clamped += 1
        row = {
            "id": rec.get("id") if fmt == SCHEMA and isinstance(rec.get("id"), str) else f"s{len(rows) + 1:04d}",
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "text": text,
            "speaker": _speaker(rec),
            "conf": _conf(rec),
            "source": str(rec.get("source") or "legacy"),
            "flags": _flags(rec),
        }
        rows.append(row)

    if records and not rows and skipped_untimed:
        raise AdaptError("E_NO_TIMING",
                         f"{skipped_untimed} записей без таймкодов (unit-границы не переданы)")

    rows.sort(key=lambda r: (r["start"], r["end"]))
    seen: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        if fmt != SCHEMA:
            row["id"] = f"s{index:04d}"
        else:
            base = row["id"] or f"s{index:04d}"
            count = seen.get(base, 0)
            seen[base] = count + 1
            row["id"] = base if count == 0 else f"{base}-{count}"

    duration_s = duration
    if duration_s is None and rows:
        duration_s = max(r["end"] for r in rows)

    payload_meta = {"vad": None, "window_s": None, "source_format": fmt,
                    "converted": len(rows), "skipped": skipped_bad, "skipped_untimed": skipped_untimed,
                    "clamped": clamped}
    for source in (src_meta, meta):
        if isinstance(source, dict):
            for key, value in source.items():
                payload_meta.setdefault(key, value)
    payload_meta["source_format"] = fmt

    out = {
        "schema": SCHEMA,
        "audio": _audio_block(src_audio or audio, duration_s),
        "engine": _engine_block(src_engine or engine),
        "segments": rows,
        "meta": payload_meta,
    }
    _validate(out)
    return out


def _validate(doc: dict) -> None:
    duration = (doc.get("audio") or {}).get("duration_s")
    last_start = None
    seen_ids = set()
    for index, row in enumerate(doc["segments"]):
        if row["start"] >= row["end"]:
            raise AdaptError("E_INVARIANT", f"сегмент {index}: start >= end ({row['start']}, {row['end']})")
        if last_start is not None and row["start"] < last_start - 1e-6:
            raise AdaptError("E_INVARIANT", f"сегмент {index}: нарушена монотонность start")
        last_start = row["start"]
        if not str(row["text"]).strip():
            raise AdaptError("E_INVARIANT", f"сегмент {index}: пустой text")
        if row["id"] in seen_ids:
            raise AdaptError("E_INVARIANT", f"повтор id {row['id']}")
        seen_ids.add(row["id"])
        if duration is not None and row["end"] > float(duration) + 0.1:
            raise AdaptError("E_INVARIANT", f"сегмент {index}: end > duration_s + 0.1")


def convert_file(src, dst=None, *, audio=None, duration_s=None, sr=None, channels=None,
                 engine=None, language=None, **kwargs) -> dict:
    """Read `src`, convert, optionally write `dst`, return the payload."""
    path = Path(src).expanduser()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AdaptError("E_INPUT", f"нет файла {path}") from exc
    except ValueError as exc:
        raise AdaptError("E_JSON", f"{path}: {exc}") from exc

    if isinstance(audio, dict):
        block = dict(audio)
    else:
        block = {"path": None}
        if isinstance(audio, (str, Path)):
            audio_path = Path(audio).expanduser()
            block["path"] = str(audio_path.resolve())
            if audio_path.exists():
                block["sha256"] = _sha256_file(audio_path)
            if duration_s is None:
                duration_s = probe_duration(audio_path)
    if block.get("duration_s") is None and duration_s is not None:
        block["duration_s"] = float(duration_s)
    if block.get("sr") is None and sr is not None:
        block["sr"] = int(sr)
    if block.get("channels") is None and channels is not None:
        block["channels"] = int(channels)

    if isinstance(engine, dict):
        engine_block = dict(engine)
        if language:
            params = dict(engine_block.get("params") or {})
            params.setdefault("language", language)
            engine_block["params"] = params
    elif language:
        engine_block = {"name": "legacy", "params": {"language": language}}
    else:
        engine_block = None

    doc = convert(data, audio=block, engine=engine_block, **kwargs)
    if dst is not None:
        out = Path(dst).expanduser()
        if out.is_dir():
            out = out / "segments.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return doc


def probe_duration(path: Path):
    """Duration in seconds via ffprobe, then soundfile, else None."""
    import subprocess

    try:
        cp = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            capture_output=True, text=True, timeout=30)
        if cp.returncode == 0 and cp.stdout.strip():
            return round(float(cp.stdout.strip()), 2)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        import soundfile as sf
        return round(float(sf.info(str(path)).duration), 2)
    except Exception:
        return None


# ---------------------------------------------------------------------------- CLI

class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse would exit 2; the contract wants 64
        print(json.dumps({"error": {"code": "E_ARGS", "detail": message}, "exit_code": 64},
                         ensure_ascii=False))
        raise SystemExit(64)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(description="Конвертер старых форматов транскриптов в asr.segments/1")
    p.add_argument("--in", "--input", dest="src", required=True, help="входной JSON")
    p.add_argument("--out", required=True, help="куда писать asr.segments/1")
    p.add_argument("--audio", help="аудио, к которому относятся сегменты")
    p.add_argument("--duration", type=float, help="длительность аудио, с")
    p.add_argument("--sr", type=int, help="частота дискретизации")
    p.add_argument("--channels", type=int, help="число каналов")
    p.add_argument("--engine", default="legacy", help="имя движка для engine-блока")
    p.add_argument("--model", help="модель движка")
    p.add_argument("--version", help="версия движка")
    p.add_argument("--language", help="язык распознавания")
    p.add_argument("--level-sensitive", action="store_true", default=False,
                   help="движок чувствителен к уровню записи")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        doc = convert_file(
            args.src, args.out,
            audio=args.audio,
            duration_s=args.duration,
            sr=args.sr,
            channels=args.channels,
            engine={"name": args.engine, "model": args.model, "version": args.version,
                    "params": {}, "level_sensitive": bool(args.level_sensitive)},
            language=args.language,
        )
    except AdaptError as exc:
        print(json.dumps({"error": {"code": exc.code, "detail": exc.detail}, "exit_code": 1},
                         ensure_ascii=False))
        return 1
    out = Path(args.out).expanduser()
    if out.is_dir():
        out = out / "segments.json"
    print(json.dumps({"schema": doc["schema"], "out": str(out), "format": doc["meta"]["source_format"],
                      "segments": len(doc["segments"]), "skipped": doc["meta"]["skipped"],
                      "skipped_untimed": doc["meta"]["skipped_untimed"],
                      "duration_s": doc["audio"]["duration_s"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
