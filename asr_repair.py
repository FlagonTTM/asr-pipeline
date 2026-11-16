#!/usr/bin/env python3
"""Targeted repair of ASR coverage holes: plan -> decode -> clip-splice merge.

Three subcommands, all reading and writing `asr.*` JSON documents:

  plan   --trust trust.json --route route.json --out DIR
         Builds `asr.repair.plan/1` from `trust.uncovered_all` only. Holes are
         the union gaps >= 1.0 s; each is padded by 0.3 s, gaps <= 1.0 s between
         padded holes are merged, anything shorter than 1.5 s is dropped. Budget
         is 0.35 * duration.

  run    --audio rec.wav --plan plan.json --engines engines.json --out DIR
         Cuts each interval with ffmpeg (-ss start-0.15, -t len+0.3, 50 ms fades,
         16 kHz mono pcm_s16le), calls the engine command from engines.json,
         shifts the returned timestamps back to global time and clips them to
         the hole. Writes attempt-N.json, attempt-N.segments.json and the
         pipeline alias additions.json. Plans outside budget are blocked (exit 3)
         unless --force-budget explicitly permits decoding.

  merge  --base whisper.json --additions attempt-1.segments.json --out merged.json
         Clip-splice: additions are clipped to hole boundaries, base segments
         overlapping a hole are trimmed at it ("whole segment inside the hole"
         inserts nothing on real data - that rule is why this is a splice).
         Invariants M1-M6 are checked; any violation raises MERGE_REJECTED and
         the untouched base is written instead (exit 2: partial, base intact).

Usage:
  asr_repair.py plan  --trust 20_gate/trust.json --route 20_gate/route.json --out 30_repair
  asr_repair.py run   --audio rec.wav --plan 30_repair/repair_plan.json \\
                      --engines engines.json --out 30_repair
  asr_repair.py merge --base fixtures/whisper.json \\
                      --additions 30_repair/attempt-1.segments.json --out merged.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

MIN_HOLE_S = 1.0          # a union gap must be at least this long to be repaired
PAD_S = 0.3               # padding around each hole
MERGE_GAP_S = 1.0         # padded holes closer than this are merged
MIN_INTERVAL_S = 1.5      # resulting intervals shorter than this are dropped
BUDGET_FRACTION = 0.35
CUT_LEAD_S = 0.15         # ffmpeg -ss start-0.15
CUT_TAIL_S = 0.15         # ffmpeg -t len+0.3
FADE_S = 0.05
DEFAULT_LANGUAGE = "ru"

MAX_OVERLAP_S = 0.2       # M1
HOLE_TOL_S = 0.05         # M2
NGRAM = 6                 # M3
NGRAM_WINDOW_S = 2.0      # M3
MIN_SEG_S = 0.03

ENGINE_NAME = "whisper"


class RepairError(Exception):
    """Failure with a stable machine-readable code in the message."""

    code = "REPAIR_ERROR"

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


class MergeRejected(RepairError):
    code = "MERGE_REJECTED"

    def __init__(self, violations: list[str], detail: str = ""):
        self.violations = violations
        super().__init__("MERGE_REJECTED", detail or "; ".join(violations))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RepairError("INPUT_MISSING", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise RepairError("INPUT_INVALID", f"{path}: {exc}") from exc


def write_json(path: Path, doc: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def merge_intervals(intervals: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    """Union of intervals, merging anything separated by <= gap."""
    rows = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    if not rows:
        return []
    out = [list(rows[0])]
    for a, b in rows[1:]:
        if a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def union_length(segments: list[dict]) -> float:
    return sum(b - a for a, b in merge_intervals([(s["start"], s["end"]) for s in segments]))


def tokens(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", str(text).lower())


def audio_duration(path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        pass
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(res.stdout.strip())
    except Exception as exc:
        raise RepairError("AUDIO_UNREADABLE", f"{path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def build_plan(trust: dict, route: dict | None = None) -> dict:
    """`asr.repair.plan/1` from union holes only (`uncovered_all`).

    Route intervals are deliberately ignored: the gate's route may name broader
    windows, but repair must re-decode exactly what the union does not cover.
    """
    metrics = trust.get("audio_metrics") or {}
    duration = metrics.get("duration_s")
    if duration is None:
        duration = (trust.get("audio") or {}).get("duration_s")
    duration = float(duration or 0.0)
    if duration <= 0:
        raise RepairError("PLAN_NO_DURATION", "trust.audio_metrics.duration_s missing")

    raw = []
    for iv in trust.get("uncovered_all") or []:
        try:
            start, end = float(iv["start"]), float(iv["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RepairError("PLAN_BAD_INTERVAL", f"{iv!r}") from exc
        if end - start >= MIN_HOLE_S:
            raw.append((max(0.0, start), min(duration, end)))

    padded = [(max(0.0, a - PAD_S), min(duration, b + PAD_S)) for a, b in raw]
    merged = merge_intervals(padded, gap=MERGE_GAP_S)

    engines = ["whisper"]
    max_attempts = 2
    budget_s = round(BUDGET_FRACTION * duration, 1)
    if route:
        budget = route.get("budget") or {}
        if isinstance(budget.get("seconds_max"), (int, float)):
            budget_s = round(float(budget["seconds_max"]), 1)
        if isinstance(budget.get("attempts_max"), int) and budget["attempts_max"] > 0:
            max_attempts = int(budget["attempts_max"])
        for step in route.get("steps") or []:
            if step.get("kind") == "repair":
                params = step.get("params") or {}
                if isinstance(params.get("engines"), list) and params["engines"]:
                    engines = [str(e) for e in params["engines"]]

    intervals = []
    for start, end in merged:
        if end - start < MIN_INTERVAL_S:
            continue
        intervals.append({
            "id": f"r{len(intervals) + 1}",
            "start": round(start, 2),
            "end": round(end, 2),
            "pad_s": PAD_S,
            "reason": "gap",
            "engines": list(engines),
            # decode cost = padded span + the 0.3 s of extra audio ffmpeg cuts
            "decode_s": round((end - start) + (CUT_LEAD_S + CUT_TAIL_S), 1),
        })

    total = round(sum(iv["decode_s"] for iv in intervals), 1)
    notes = []
    if not intervals:
        notes.append("no union gaps >= 1.0 s")
    if total > budget_s:
        notes.append(f"plan needs {total} s > budget {budget_s} s")

    return {
        "schema": "asr.repair.plan/1",
        "attempt": 1,
        "intervals": intervals,
        "total_decode_s": total,
        "budget_s": budget_s,
        "within_budget": total <= budget_s + 1e-9,
        "stop_rule": {"min_gain_ratio": 0.25, "max_attempts": max_attempts},
        "duration_s": round(duration, 2),
        "notes": notes,
    }


def cmd_plan(args: argparse.Namespace) -> int:
    trust = load_json(Path(args.trust))
    route = load_json(Path(args.route)) if args.route else None
    plan = build_plan(trust, route)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "repair_plan.json", plan)
    write_json(out / "plan.json", plan)  # stable alias for pipeline stages
    print(json.dumps({"ok": True, "schema": plan["schema"], "intervals": len(plan["intervals"]),
                      "total_decode_s": plan["total_decode_s"], "budget_s": plan["budget_s"],
                      "within_budget": plan["within_budget"],
                      "out": str(out / "repair_plan.json")}, ensure_ascii=False))
    return 0 if plan["within_budget"] else 2


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cut_interval(audio: Path, start: float, end: float, out_path: Path,
                 duration: float | None = None) -> dict:
    """Cut [start-0.15, end+0.15] as 16 kHz mono pcm_s16le with 50 ms fades."""
    cut_start = max(0.0, start - CUT_LEAD_S)
    cut_end = end + CUT_TAIL_S
    if duration:
        cut_end = min(duration, cut_end)
    cut_len = max(0.0, cut_end - cut_start)
    if cut_len < 0.1:
        raise RepairError("CUT_TOO_SHORT", f"{start}-{end}: {cut_len:.3f} s")
    fade_out_start = max(0.0, cut_len - FADE_S)
    afade = (f"afade=t=in:st=0:d={FADE_S},"
             f"afade=t=out:st={fade_out_start:.3f}:d={FADE_S}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", f"{cut_start:.3f}", "-t", f"{cut_len:.3f}",
        "-i", str(audio), "-vn", "-af", afade,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RepairError("CUT_FAILED", (res.stderr or "").strip()[-400:])
    return {"path": str(out_path), "offset_s": round(cut_start, 3), "dur_s": round(cut_len, 3)}


def engine_command(cfg: dict, audio: Path, out: Path, language: str) -> list[str]:
    template = cfg.get("cmd")
    if not template:
        raise RepairError("ENGINE_NO_CMD", "engines.json entry has no 'cmd'")
    parts = shlex.split(template) if isinstance(template, str) else [str(p) for p in template]
    import re as _re

    def _env(text: str) -> str:
        def _sub(m):
            name, default = m.group(1), m.group(2)
            if name == "ASR_PY_ROOT" and name not in os.environ:
                return str(Path(__file__).resolve().parent)
            return os.environ.get(name, default if default is not None else "")
        return _re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", _sub, text)

    parts = [_env(p) for p in parts]
    mapping = {"{audio}": str(audio), "{out}": str(out), "{language}": language,
               "{model}": str(cfg.get("model", ""))}
    cmd = []
    for part in parts:
        for key, value in mapping.items():
            part = part.replace(key, value)
        cmd.append(part)
    return cmd


def run_attempt(audio: Path, plan: dict, engines_cfg: dict, out_dir: Path,
                language: str = DEFAULT_LANGUAGE) -> dict:
    audio = Path(audio).resolve()
    out_dir = Path(out_dir).resolve()
    attempt = int(plan.get("attempt") or 1)
    (out_dir / "engine").mkdir(parents=True, exist_ok=True)
    duration = audio_duration(audio)
    limits = plan.get("budget_s")
    spent = 0.0
    all_segments: list[dict] = []
    results: list[dict] = []
    used_engines: list[str] = []
    holes: list[dict] = []

    for iv in plan.get("intervals") or []:
        iv_id = str(iv.get("id") or f"r{len(holes) + 1}")
        start, end = float(iv["start"]), float(iv["end"])
        holes.append({"id": iv_id, "start": round(start, 3), "end": round(end, 3)})
        names = [str(e) for e in (iv.get("engines") or [ENGINE_NAME])]
        for name in names:
            cfg = engines_cfg.get(name)
            if not cfg:
                raise RepairError("ENGINE_UNKNOWN", f"{name} not in engines config")
            if name not in used_engines:
                used_engines.append(name)
            cut_path = out_dir / "cuts" / f"{iv_id}.{name}.wav"
            eng_dir = out_dir / "engine" / f"{iv_id}.{name}"
            record = {"id": iv_id, "start": round(start, 3), "end": round(end, 3),
                      "engine": name, "status": "error", "rc": None, "cut": None,
                      "offset_s": None, "decode_s": 0.0, "segments": 0, "stderr_tail": ""}
            try:
                cut = cut_interval(audio, start, end, cut_path, duration)
                record["cut"] = str(cut["path"])
                record["offset_s"] = cut["offset_s"]
                record["decode_s"] = cut["dur_s"]
                spent += cut["dur_s"]
                cmd = engine_command(cfg, cut_path, eng_dir, language)
                res = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=float(cfg.get("timeout_s") or 900))
                record["rc"] = res.returncode
                record["stderr_tail"] = (res.stderr or "").strip()[-400:]
                if res.returncode != 0:
                    raise RepairError("ENGINE_FAILED", f"{name} rc={res.returncode}")
                produced = eng_dir / (cfg.get("output") or "whisper.segments.json")
                if not produced.exists():
                    found = sorted(p for p in eng_dir.glob("*.json") if p.is_file())
                    if not found:
                        raise RepairError("ENGINE_BAD_OUTPUT", f"{eng_dir}: no json")
                    produced = max(found, key=lambda p: p.stat().st_size)
                doc = load_json(produced)
                rows = doc.get("segments")
                if not isinstance(rows, list):
                    raise RepairError("ENGINE_BAD_OUTPUT", f"{eng_out}: no segments")
                added = _place_segments(rows, start, end, cut["offset_s"], name, iv_id, len(all_segments))
                all_segments.extend(added)
                record["segments"] = len(added)
                record["status"] = "ok"
                record["engine_output"] = str(produced)
            except (RepairError, subprocess.TimeoutExpired, OSError) as exc:
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                if isinstance(exc, subprocess.TimeoutExpired):
                    record["rc"] = -9
            results.append(record)

    all_segments.sort(key=lambda s: (s["start"], s["end"]))
    for i, seg in enumerate(all_segments, 1):
        seg["id"] = f"s{i:04d}"

    segments_doc = {
        "schema": "asr.segments/1",
        "audio": {"path": str(audio), "sha256": sha256_file(audio),
                  "duration_s": round(duration, 2), "sr": 16000, "channels": 1},
        "engine": {"name": "+".join(used_engines) or ENGINE_NAME, "model": "",
                   "version": "repair", "params": {"language": language, "fade_ms": 50},
                   "level_sensitive": True},
        "segments": all_segments,
        "meta": {"vad": "off", "window_s": None,
                 "repair": {"attempt": attempt, "plan_attempt": plan.get("attempt"),
                            "intervals": holes}},
    }
    segments_path = out_dir / f"attempt-{attempt}.segments.json"
    write_json(segments_path, segments_doc)
    # pipeline alias: 40_repair/additions.json always holds the latest attempt
    write_json(out_dir / "additions.json", segments_doc)

    failed = [r for r in results if r["status"] != "ok"]
    attempt_doc = {
        "schema": "asr.repair.attempt/1",
        "attempt": attempt,
        "audio": {"path": str(audio), "sha256": sha256_file(audio), "duration_s": round(duration, 2)},
        "intervals": results,
        "segments_n": len(all_segments),
        "total_decode_s": round(spent, 1),
        "budget_s": limits,
        "status": "ok" if not failed else ("partial" if len(failed) < len(results) else "failed"),
        "segments_path": segments_path.name,
    }
    write_json(out_dir / f"attempt-{attempt}.json", attempt_doc)
    return attempt_doc


def _place_segments(rows: list[dict], hole_start: float, hole_end: float, offset: float,
                    engine: str, interval_id: str, base_index: int) -> list[dict]:
    """Shift engine-local times to global time and clip them to the hole."""
    out = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        try:
            local_start, local_end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start = max(hole_start, offset + local_start)
        end = min(hole_end, offset + local_end)
        if end - start < MIN_SEG_S:
            continue
        out.append({
            "id": f"{interval_id}-{len(out) + base_index + 1:03d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "speaker": None,
            "conf": row.get("conf"),
            "source": f"repair:{engine}:{interval_id}",
            "flags": ["clipped"],
        })
    return out


def cmd_run(args: argparse.Namespace) -> int:
    audio = Path(args.audio)
    if not audio.is_file():
        raise RepairError("AUDIO_MISSING", str(audio))
    plan = load_json(Path(args.plan))
    if plan.get("within_budget") is False and not args.force_budget:
        print(json.dumps({"status": "blocked", "code": "REPAIR_BUDGET_EXCEEDED",
                          "planned_decode_s": plan.get("total_decode_s"),
                          "budget_s": plan.get("budget_s")}, ensure_ascii=False))
        return 3
    engines_cfg = load_json(Path(args.engines))
    doc = run_attempt(audio, plan, engines_cfg, Path(args.out), args.language)
    print(json.dumps({"ok": doc["status"] != "failed", "status": doc["status"],
                      "attempt": doc["attempt"], "segments": doc["segments_n"],
                      "total_decode_s": doc["total_decode_s"],
                      "out": str(Path(args.out) / f"attempt-{doc['attempt']}.json")},
                     ensure_ascii=False))
    return 0 if doc["status"] == "ok" else 2


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def _norm(seg: dict, source: str = "primary") -> dict:
    return {
        "id": str(seg.get("id") or ""),
        "start": round(float(seg["start"]), 3),
        "end": round(float(seg["end"]), 3),
        "text": str(seg.get("text") or "").strip(),
        "speaker": seg.get("speaker"),
        "conf": seg.get("conf"),
        "source": str(seg.get("source") or source),
        "flags": list(seg.get("flags") or []),
    }


def _is_repair(seg: dict) -> bool:
    return str(seg.get("source") or "").startswith("repair:")


def collect_holes(additions: dict) -> list[dict]:
    """Hole boundaries the additions were decoded for.

    Preferred source is `meta.repair.intervals` written by `run`. A peer segment
    file without that block falls back to the union of the additions themselves,
    which makes the splice a no-op instead of a guess.
    """
    meta = ((additions.get("meta") or {}).get("repair") or {})
    raw = meta.get("intervals") or meta.get("holes") or []
    holes = []
    for h in raw:
        try:
            start, end = float(h["start"]), float(h["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            holes.append({"id": str(h.get("id") or f"h{len(holes) + 1}"),
                          "start": round(start, 3), "end": round(end, 3)})
    holes.sort(key=lambda h: h["start"])
    if holes:
        return holes
    segs = [_norm(s) for s in (additions.get("segments") or []) if float(s.get("end", 0)) > float(s.get("start", 0))]
    return [{"id": f"h{i + 1}", "start": round(a, 3), "end": round(b, 3)}
            for i, (a, b) in enumerate(merge_intervals([(s["start"], s["end"]) for s in segs]))]


def _clip_additions(additions: dict, holes: list[dict]) -> tuple[list[dict], list[str], int]:
    """Clip every addition to its hole; returns (segments, violations, dropped)."""
    out, violations, dropped = [], [], 0
    for seg in additions.get("segments") or []:
        try:
            row = _norm(seg)
        except (KeyError, TypeError, ValueError):
            violations.append(f"M2: malformed addition {seg!r}")
            continue
        if not row["text"]:
            dropped += 1
            continue
        best, best_overlap = None, 0.0
        for hole in holes:
            overlap = min(row["end"], hole["end"]) - max(row["start"], hole["start"])
            if overlap > best_overlap:
                best, best_overlap = hole, overlap
        if best is None or best_overlap <= 0.0:
            violations.append(f"M2: segment {row['start']}-{row['end']} lies outside every planned interval")
            continue
        start = max(row["start"], best["start"])
        end = min(row["end"], best["end"])
        if end - start < MIN_SEG_S:
            dropped += 1
            continue
        row["start"], row["end"] = round(start, 3), round(end, 3)
        row["flags"] = sorted(set(row["flags"]) | {"clipped"})
        out.append(row)
    return out, violations, dropped


def _trim_base(base_segs: list[dict], holes: list[dict]) -> tuple[list[dict], int]:
    """Cut base segments at hole boundaries; split when a segment spans a hole."""
    out, trimmed = [], 0
    for seg in base_segs:
        pieces = [(seg["start"], seg["end"])]
        for hole in holes:
            nxt = []
            for a, b in pieces:
                if b <= hole["start"] or a >= hole["end"]:
                    nxt.append((a, b))
                    continue
                trimmed += 1
                if a < hole["start"]:
                    nxt.append((a, hole["start"]))
                if b > hole["end"]:
                    nxt.append((hole["end"], b))
            pieces = nxt
        for a, b in pieces:
            if b - a < MIN_SEG_S:
                continue
            row = copy.deepcopy(seg)
            row["start"], row["end"] = round(a, 3), round(b, 3)
            row["flags"] = sorted(set(row["flags"]) | {"trimmed"}) if (a, b) != (seg["start"], seg["end"]) else row["flags"]
            out.append(row)
    return out, trimmed


def _resolve(segments: list[dict]) -> list[dict]:
    """Sort and remove overlaps, preferring repair segments over base ones."""
    rows = sorted(segments, key=lambda s: (s["start"], 0 if _is_repair(s) else 1, -s["end"]))
    out: list[dict] = []
    for seg in rows:
        if seg["end"] - seg["start"] < MIN_SEG_S or not seg["text"]:
            continue
        if out and seg["start"] < out[-1]["end"] - 1e-9:
            prev = out[-1]
            if _is_repair(seg) and not _is_repair(prev):
                prev["end"] = round(seg["start"], 3)
            else:
                seg["start"] = round(prev["end"], 3)
            if prev["end"] - prev["start"] < MIN_SEG_S:
                out.pop()
            if seg["end"] - seg["start"] < MIN_SEG_S:
                continue
        if out and abs(out[-1]["start"] - seg["start"]) < 1e-9 and abs(out[-1]["end"] - seg["end"]) < 1e-9:
            out[-1]["text"] = out[-1]["text"] or seg["text"]
            continue
        out.append(seg)
    return out


def _ngram_repeats(segments: list[dict], window: float = NGRAM_WINDOW_S) -> list[str]:
    """M3: a 6-gram repeated within `window` seconds, where a repair is involved."""
    seen: dict[tuple, list[tuple[float, bool]]] = {}
    for seg in segments:
        toks = tokens(seg["text"])
        if len(toks) < NGRAM:
            continue
        span = max(1e-6, seg["end"] - seg["start"])
        times = [seg["start"] + span * (i / max(1, len(toks) - 1)) for i in range(len(toks))]
        for i in range(len(toks) - NGRAM + 1):
            seen.setdefault(tuple(toks[i:i + NGRAM]), []).append((times[i], _is_repair(seg)))
    bad = []
    for gram, hits in seen.items():
        if len(hits) < 2:
            continue
        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                if abs(hits[i][0] - hits[j][0]) <= window and (hits[i][1] or hits[j][1]):
                    bad.append(f"M3: {' '.join(gram)} repeated within {window:.1f} s")
                    break
            else:
                continue
            break
        if len(bad) >= 5:
            break
    return bad


def check_invariants(base_segs: list[dict], merged: list[dict], holes: list[dict]) -> list[str]:
    violations: list[str] = []
    # M1: no overlap > 0.2 s
    for a, b in zip(merged, merged[1:]):
        overlap = a["end"] - b["start"]
        if overlap > MAX_OVERLAP_S:
            violations.append(f"M1: overlap {overlap:.2f} s at {b['start']:.2f}")
    # M2: every repair segment inside a planned interval +-0.05 s
    for seg in merged:
        if not _is_repair(seg):
            continue
        inside = any(seg["start"] >= h["start"] - HOLE_TOL_S and seg["end"] <= h["end"] + HOLE_TOL_S
                     for h in holes)
        if not inside:
            violations.append(f"M2: repair segment {seg['start']}-{seg['end']} outside planned intervals")
    # M3: no 6-gram repeated within 2 s
    violations += _ngram_repeats(merged)
    # M4: union coverage must grow
    before, after = union_length(base_segs), union_length(merged)
    if after <= before + 1e-6:
        violations.append(f"M4: union coverage did not grow ({before:.2f} -> {after:.2f} s)")
    # M5: at most 2x the base segment count
    if len(merged) > max(1, 2 * len(base_segs)):
        violations.append(f"M5: {len(merged)} segments > 2x base ({len(base_segs)})")
    # M6: monotonic, non-empty timestamps
    last = -1.0
    for seg in merged:
        if not seg["start"] < seg["end"]:
            violations.append(f"M6: bad span {seg['start']}-{seg['end']}")
        if seg["start"] < last - 1e-9:
            violations.append(f"M6: timestamps not monotonic at {seg['start']:.2f}")
        last = seg["start"]
    return violations


def merge_documents(base: dict, additions: dict) -> dict:
    """Clip-splice additions into base; raises MergeRejected on M1-M6 failure."""
    base_doc = copy.deepcopy(base)
    base_segs = []
    for seg in base.get("segments") or []:
        try:
            base_segs.append(_norm(seg))
        except (KeyError, TypeError, ValueError) as exc:
            raise RepairError("MERGE_BAD_BASE", f"{seg!r}") from exc
    holes = collect_holes(additions)
    if not holes:
        raise MergeRejected(["M2: additions carry no planned intervals"])

    add_segs, violations, dropped = _clip_additions(additions, holes)
    trimmed_segs, trimmed = _trim_base(base_segs, holes)
    merged = _resolve(trimmed_segs + add_segs)

    # base order must already be monotonic; keep original ids stable
    used = set()
    remaining = {s["id"] for s in merged if not _is_repair(s) and s["id"]}
    for seg in merged:
        cid = seg["id"]
        is_rep = _is_repair(seg)
        if not is_rep:
            remaining.discard(cid)
        if cid and cid not in used and not (is_rep and cid in remaining):
            used.add(cid)
            continue
        prefix = "rp" if is_rep else "s"
        n = 1
        while True:  # never hand out a duplicate id, nor steal a later base id
            candidate = f"{prefix}{n:04d}"
            if candidate not in used and candidate not in remaining:
                break
            n += 1
        seg["id"] = candidate
        used.add(candidate)

    violations += check_invariants(base_segs, merged, holes)
    if violations:
        raise MergeRejected(violations)

    before, after = union_length(base_segs), union_length(merged)
    planned = sum(h["end"] - h["start"] for h in holes)
    repair_meta = ((additions.get("meta") or {}).get("repair") or {})
    base_doc["segments"] = merged
    meta = base_doc.setdefault("meta", {})
    meta["repair"] = {
        "attempt": repair_meta.get("attempt"),
        "intervals": holes,
        "added": sum(1 for s in merged if _is_repair(s)),
        "dropped": dropped,
        "trimmed": trimmed,
        "coverage_before_s": round(before, 3),
        "coverage_after_s": round(after, 3),
        "gain_s": round(after - before, 3),
        "gain_ratio": round((after - before) / planned, 3) if planned > 0 else 0.0,
        "invariants": {"M1": "ok", "M2": "ok", "M3": "ok", "M4": "ok", "M5": "ok", "M6": "ok"},
    }
    if additions.get("audio", {}).get("sha256") and base.get("audio", {}).get("sha256") \
            and additions["audio"]["sha256"] != base["audio"]["sha256"]:
        meta["repair"]["audio_mismatch"] = True
    return base_doc


def cmd_merge(args: argparse.Namespace) -> int:
    base = load_json(Path(args.base))
    additions = load_json(Path(args.additions))
    out = Path(args.out)
    try:
        doc = merge_documents(base, additions)
    except MergeRejected as exc:
        write_json(out, base)  # contract: on rejection the base comes back unchanged
        print(json.dumps({"ok": False, "code": exc.code, "violations": exc.violations,
                          "segments": len(base.get("segments") or []), "out": str(out)},
                         ensure_ascii=False))
        print(exc, file=sys.stderr)
        return 2  # partial: the base is intact, repair simply did not land
    write_json(out, doc)
    rep = doc["meta"]["repair"]
    print(json.dumps({"ok": True, "segments": len(doc["segments"]), "added": rep["added"],
                      "coverage_before_s": rep["coverage_before_s"],
                      "coverage_after_s": rep["coverage_after_s"], "out": str(out)},
                     ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    class Parser(argparse.ArgumentParser):
        def error(self, message: str):  # contract: bad arguments -> 64
            self.print_usage(sys.stderr)
            print(json.dumps({"ok": False, "code": "BAD_ARGS", "detail": message},
                             ensure_ascii=False))
            raise SystemExit(64)

    p = Parser(description="ASR repair: plan / run / merge")
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("plan", help="build asr.repair.plan/1 from union holes")
    pp.add_argument("--trust", required=True)
    pp.add_argument("--route", default=None)
    pp.add_argument("--out", required=True)
    pp.set_defaults(func=cmd_plan)

    pr = sub.add_parser("run", help="cut intervals and decode them with engines.json")
    pr.add_argument("--audio", required=True)
    pr.add_argument("--plan", required=True)
    pr.add_argument("--engines", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--language", default=DEFAULT_LANGUAGE)
    pr.add_argument("--force-budget", action="store_true",
                    help="explicitly allow decoding a plan that exceeds its budget")
    pr.set_defaults(func=cmd_run)

    pm = sub.add_parser("merge", help="clip-splice additions into base")
    pm.add_argument("--base", required=True)
    pm.add_argument("--additions", required=True)
    pm.add_argument("--out", required=True)
    pm.set_defaults(func=cmd_merge)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RepairError as exc:
        print(json.dumps({"ok": False, "code": exc.code, "detail": exc.detail}, ensure_ascii=False))
        print(exc, file=sys.stderr)
        return 2 if exc.code == "MERGE_REJECTED" else 1
    except Exception as exc:  # noqa: BLE001 - CLI must always answer with JSON
        print(json.dumps({"ok": False, "code": "INTERNAL", "detail": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
