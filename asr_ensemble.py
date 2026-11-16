#!/usr/bin/env python3
"""Equal-weight medoid ensemble over fixed windows.

For every window, each engine's hypothesis is scored by its mean token Dice
against the other engines; the medoid wins. This is the x20 selection rule
without the progressive confusion network: cheap, deterministic, and it keeps
the provenance of every segment.

Usage:
  asr_ensemble.py --segments gigaam=a.json --segments whisper=b.json \
                  --segments parakeet=c.json --segments qwen=d.json \
                  --out ensemble/ [--window 20]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import soundfile as sf


def toks(text: str) -> set[str]:
    return set(re.findall(r"[а-яёa-z0-9]+", (text or "").lower()))


def dice(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def window_text(segments: list[dict], lo: float, hi: float, *, window_s: float | None = None) -> str:
    """Keep each whole segment in exactly one window of the zero-based grid.

    Maximum overlap wins; ties go to the earlier window. This also assigns
    segments when no overlap reaches 50% of their duration or two seconds,
    without duplicating them when multiple windows pass those thresholds.
    Supply the original grid width for a shortened final window.
    """
    width = window_s if window_s is not None else hi - lo
    texts = []
    for s in segments:
        start, end = float(s.get("start") or 0), float(s.get("end") or 0)
        if end <= start or end <= lo or start >= hi:
            continue
        first = max(0, math.floor(start / width))
        last = math.ceil(end / width) - 1
        owner = max(range(first, last + 1),
                    key=lambda i: min(end, (i + 1) * width) - max(start, i * width))
        if owner * width == lo:
            texts.append(str(s.get("text") or ""))
    return " ".join(texts).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--audio", default=None)
    a = ap.parse_args()

    engines: dict[str, list[dict]] = {}
    for spec in a.segments:
        name, _, path = spec.partition("=")
        engines[name] = json.loads(Path(path).read_text(encoding="utf-8")).get("segments") or []
    if not engines:
        raise SystemExit("no engines")

    first = next(iter(engines.values()))
    duration = max((float(s.get("end") or 0) for segs in engines.values() for s in segs), default=0.0)
    if a.audio:
        info = sf.info(a.audio)
        duration = max(duration, float(info.duration))
    n = int(-(-duration // a.window))

    raw_confidences = []
    out_segments, report = [], {"schema": "asr.ensemble/1", "window_s": a.window,
                                "engines": list(engines), "windows": []}
    for i in range(n):
        lo, hi = i * a.window, min((i + 1) * a.window, duration)
        cand = {name: window_text(segs, lo, hi, window_s=a.window) for name, segs in engines.items()}
        uniq = {name: text for name, text in cand.items() if text}
        if not uniq:
            continue
        single_source = len(uniq) == 1
        scores = {}
        for name, text in uniq.items():
            others = [toks(t) for n2, t in uniq.items() if n2 != name]
            mine = toks(text)
            scores[name] = sum(dice(mine, o) for o in others) / len(others) if others else 0.0
        best = max(uniq, key=lambda name: (scores.get(name, 0.0), -list(engines).index(name)))
        raw_confidences.append(scores[best])
        out_segments.append({"id": f"w{i:03d}", "start": round(lo, 2), "end": round(hi, 2),
                             "text": uniq[best], "speaker": None, "conf": round(scores[best], 3),
                             "source": best, "flags": ["single_source"] if single_source else []})
        report["windows"].append({"start": round(lo, 2), "end": round(hi, 2), "chosen": best,
                                  "scores": {k: round(v, 3) for k, v in scores.items()},
                                  "empty": [k for k, t in cand.items() if not t],
                                  "single_source": single_source})
    votes = {}
    for w in report["windows"]:
        votes[w["chosen"]] = votes.get(w["chosen"], 0) + 1
    report["votes"] = votes
    report["segments"] = len(out_segments)
    report["mean_confidence"] = round(sum(raw_confidences) / len(raw_confidences), 3) if raw_confidences else None

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "asr.segments/1",
           "audio": {"path": str(a.audio or ""), "sha256": "", "duration_s": round(duration, 2),
                     "sr": 16000, "channels": 1},
           "engine": {"name": "ensemble", "model": "+".join(engines), "version": "medoid", "params": {"window_s": a.window},
                      "level_sensitive": True},
           "segments": out_segments, "meta": {"vad": "off", "window_s": a.window, "votes": votes}}
    (out / "segments.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"segments": len(out_segments), "votes": votes,
                      "mean_confidence": report["mean_confidence"], "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
