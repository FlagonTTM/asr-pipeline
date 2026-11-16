#!/usr/bin/env python3
"""faster-whisper (CTranslate2) adapter: audio file -> `asr.segments/1` JSON.

The audio is decoded with soundfile and handed to faster-whisper as a float32
numpy array on purpose. Passing a *path* makes faster-whisper use PyAV, which
raises `TypeError` on this install; the array path keeps the dependency surface
at soundfile only.

Logs go to stderr, JSON goes to the --out file. Nothing is downloaded:
`local_files_only=True`, CPU + int8.

Usage:
  whisper_ct2.py --audio cut.wav --out seg.json --language ru
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

DEFAULT_MODEL = "large-v3"
DECODE_PARAMS = {
    "beam_size": 5,
    "vad_filter": False,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "best_of": 5,
    "suppress_blank": True,
}


def log(msg: str) -> None:
    print(f"[whisper_ct2] {msg}", file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="faster-whisper -> asr.segments/1")
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--language", default="ru")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--beam-size", type=int, default=DECODE_PARAMS["beam_size"])
    return p.parse_args(argv)


def load_audio(path: Path):
    """Decode to mono float32 via soundfile (never hand the path to faster-whisper)."""
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    x = np.asarray(x, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1, dtype="float32")
    return x, int(sr)


def _clean_rows(raw_rows: list[dict], duration: float) -> list[dict]:
    out: list[dict] = []
    last_start = -1.0
    for r in raw_rows:
        text = str(r.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, float(r["start"]))
        end = min(float(r["end"]), duration + 0.1)
        if not end > start:
            continue
        if start < last_start:  # keep the invariant even if the engine stutters
            start = last_start
            end = min(max(end, last_start + 0.01), duration + 0.1)
        conf = r.get("conf")
        out.append({"start": round(start, 3), "end": round(end, 3), "text": text, "conf": conf})
        last_start = start
    for i, row in enumerate(out, 1):
        row["id"] = f"s{i:04d}"
    return out


def build_doc(path: Path, x, sr: int, rows: list[dict], model: str, language: str,
              extra_meta: dict | None = None) -> dict:
    duration = len(x) / sr if sr else 0.0
    clean = _clean_rows(rows, duration)
    meta = {"vad": "off", "window_s": None, "language": language}
    if extra_meta:
        meta.update(extra_meta)
    return {
        "schema": "asr.segments/1",
        "audio": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "duration_s": round(duration, 2),
            "sr": sr,
            "channels": 1,
        },
        "engine": {
            "name": "whisper",
            "model": model,
            "version": "ct2",
            "params": {
                "device": "cpu",
                "compute_type": "int8",
                "language": language,
                "local_files_only": True,
                **DECODE_PARAMS,
            },
            "level_sensitive": True,
        },
        "segments": [
            {"id": r["id"], "start": r["start"], "end": r["end"], "text": r["text"],
             "speaker": None, "conf": r["conf"], "source": "primary", "flags": []}
            for r in clean
        ],
        "meta": meta,
    }


def transcribe(x, sr: int, model_name: str, language: str, beam_size: int) -> tuple[list[dict], dict]:
    from faster_whisper import WhisperModel

    log(f"loading {model_name} (cpu/int8, local_files_only)")
    model = WhisperModel(model_name, device="cpu", compute_type="int8", local_files_only=True)
    log(f"decoding {len(x) / sr:.2f} s, language={language}, beam={beam_size}")
    segments, info = model.transcribe(
        x,
        language=language,
        beam_size=beam_size,
        vad_filter=False,
        temperature=0.0,
        condition_on_previous_text=False,
        best_of=DECODE_PARAMS["best_of"],
        suppress_blank=DECODE_PARAMS["suppress_blank"],
    )
    rows = []
    for s in segments:
        conf = getattr(s, "avg_logprob", None)
        rows.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": str(s.text),
            "conf": round(math.exp(conf), 3) if isinstance(conf, (int, float)) else None,
        })
    meta = {
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 3),
        "duration_after_vad_s": round(float(getattr(info, "duration_after_vad", 0.0) or 0.0), 2),
    }
    log(f"got {len(rows)} segments")
    return rows, meta


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    audio = Path(a.audio)
    if not audio.is_file():
        print(json.dumps({"ok": False, "code": "EN_AUDIO_MISSING", "audio": str(audio)},
                         ensure_ascii=False), file=sys.stderr)
        return 64
    try:
        x, sr = load_audio(audio)
        peak = float(abs(x).max()) if len(x) else 0.0
        gain = (0.891 / peak) if peak > 1e-6 else 1.0
        if gain != 1.0:
            x = x * gain
            log(f"peak-normalised x{gain:.2f} (peak {20 * math.log10(peak + 1e-12):.1f} dBFS -> -1.0 dBFS)")
        rows, meta = transcribe(x, sr, a.model, a.language, a.beam_size)
        kept = [r for r in rows if len(str(r.get("text") or "").split()) >= 2
                or (float(r.get("end") or 0) - float(r.get("start") or 0)) < 1.0]
        dropped = len(rows) - len(kept)
        if dropped:
            log(f"dropped {dropped} degenerate segments (<2 words)")
        meta["dropped_degenerate"] = dropped
        doc = build_doc(audio, x, sr, kept, a.model, a.language, meta)
        doc["engine"]["params"].update({"input_gain": round(gain, 4), "peak_normalised": gain != 1.0})
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as exc:  # surface the code, never a bare traceback
        print(json.dumps({"ok": False, "code": "EN_FAILED", "detail": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), file=sys.stderr)
        log(f"failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps({"ok": True, "segments": len(doc["segments"]), "out": str(out),
                      "duration_s": doc["audio"]["duration_s"], "language": a.language},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
