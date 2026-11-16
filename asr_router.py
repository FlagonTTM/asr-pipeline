#!/usr/bin/env python3
"""Route planner: trust card (`asr.trust/2`) -> route plan (`asr.route/1`).

Pure function `plan_route(trust, cfg=None)` plus a thin CLI:

  asr_router.py --trust PATH --out PATH [--budget-frac 0.35]

The planner never decodes audio and never imports other project modules: it maps
gate reasons to an ordered list of cheap corrective steps, states what stays
unresolved and decides whether a summary may be produced at all.

Reason -> step mapping (CONTRACT.md):
  LOW_LEVEL        -> normalize (gain to -1 dBFS peak, capped at +40 dB),
                      only when the engine is level sensitive
  CLIPPING         -> no_gain; blocked when clip_fraction > 0.005
  VAD_UNRELIABLE   -> detector=energy
  LOW_COVERAGE     -> second_engine; blocked after a repeat when union < 0.65
  TAIL_UNCOVERED   -> repair_decode (intervals from reasons / uncovered_all)
  SPEECH_GAP       -> repair_decode (same interval sources)
  LOW_AGREEMENT    -> arbitrate (windows with dice < 0.35)
  LANGUAGE_MISMATCH-> blocked, summary_allowed=false
  unknown code     -> blocked UNKNOWN_REASON (fail-closed)

Step order is fixed: normalize, detector, no_gain, second_engine,
fill_from_peers, repair_decode, arbitrate, summarize. Only steps that are
actually required by the present reason codes are emitted.

Exit codes (CONTRACT.md): 0 ok, 2 partial, 3 failed, 4 blocked, 1 internal,
64 bad arguments.

cfg keys (all optional):
  budget_frac          float, default 0.35
  primary              engine name (default: trust.primary / agreement.pair[0])
  peer_engine          engine for second_engine / fill_from_peers
  level_sensitive      bool, engine gain sensitivity
  engines              {"name": {"level_sensitive": bool}}
  second_engine_done   bool, a second engine already ran (repeat exhausted)
  fill_from_peers      bool, ask for peer segments to fill holes
  summarize            bool, append the final summarize step to the route
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA = "asr.route/1"
DEFAULT_BUDGET_FRAC = 0.35
ATTEMPTS_MAX = 2
GAIN_CAP_DB = 40.0
TARGET_PEAK_DBFS = -1.0
CLIP_BLOCK_FRACTION = 0.005
MIN_UNION_AFTER_REPEAT = 0.65
MIN_HOLE_S = 1.0
DICE_BLOCK = 0.35

STEP_ORDER = (
    "normalize",
    "detector",
    "no_gain",
    "second_engine",
    "fill_from_peers",
    "repair_decode",
    "arbitrate",
    "summarize",
)
KNOWN_CODES = frozenset(
    {
        "LANGUAGE_MISMATCH",
        "LOW_LEVEL",
        "CLIPPING",
        "VAD_UNRELIABLE",
        "TAIL_UNCOVERED",
        "LOW_COVERAGE",
        "SPEECH_GAP",
        "LOW_AGREEMENT",
        "LOW_INFORMATION",
    }
)
REPAIR_CODES = frozenset({"TAIL_UNCOVERED", "SPEECH_GAP"})
BLOCK_PRIORITY = {"UNKNOWN_REASON": 0, "LANGUAGE_MISMATCH": 1, "CLIPPING": 2, "LOW_COVERAGE": 3}
VERDICT_RANK = {"ok": 0, "partial": 1, "failed": 2}
EXIT_BY_VERDICT = {"ok": 0, "partial": 2, "failed": 3}
EXIT_BLOCKED = 4
EXIT_INTERNAL = 1
EXIT_BAD_ARGS = 64


class RouteError(Exception):
    """Error with a stable machine code in the message text."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_mapping(value, what: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RouteError("ROUTE_BAD_TRUST", f"{what} должен быть объектом, получено {type(value).__name__}")
    return value


def _as_list(value, what: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RouteError("ROUTE_BAD_TRUST", f"{what} должен быть массивом, получено {type(value).__name__}")
    return value


def _reasons(trust: dict) -> list[dict]:
    out = []
    for i, raw in enumerate(_as_list(trust.get("reasons"), "reasons")):
        if isinstance(raw, str):
            raw = {"code": raw}
        if not isinstance(raw, dict):
            raise RouteError("ROUTE_BAD_TRUST", f"reasons[{i}] должен быть объектом")
        code = raw.get("code")
        if not isinstance(code, str) or not code.strip():
            raise RouteError("ROUTE_BAD_TRUST", f"reasons[{i}].code отсутствует или пуст")
        item = dict(raw)
        item["code"] = code.strip()
        if "severity" not in item or item["severity"] is None:
            item["severity"] = "block" if item["code"] == "LANGUAGE_MISMATCH" else "warn"
        out.append(item)
    return out


def _primary_name(trust: dict, cfg: dict) -> str:
    primary = cfg.get("primary") or trust.get("primary")
    if isinstance(primary, str) and primary:
        return primary
    pair = (trust.get("agreement") or {}).get("pair") if isinstance(trust.get("agreement"), dict) else None
    if isinstance(pair, list) and pair and isinstance(pair[0], str):
        return pair[0]
    coverage = trust.get("coverage") if isinstance(trust.get("coverage"), dict) else {}
    for key in coverage:
        return key
    return "primary"


def _peer_name(trust: dict, cfg: dict, primary: str) -> str:
    peer = cfg.get("peer_engine")
    if isinstance(peer, str) and peer:
        return peer
    agr = trust.get("agreement") if isinstance(trust.get("agreement"), dict) else {}
    pair = agr.get("pair")
    if isinstance(pair, list):
        for name in pair:
            if isinstance(name, str) and name and name != primary:
                return name
    coverage = trust.get("coverage") if isinstance(trust.get("coverage"), dict) else {}
    for name in coverage:
        if name != primary:
            return name
    return "second"


def _duration(trust: dict) -> float:
    metrics = trust.get("audio_metrics") if isinstance(trust.get("audio_metrics"), dict) else {}
    dur = metrics.get("duration_s")
    if _is_num(dur) and float(dur) > 0:
        return float(dur)
    return 0.0


def _level_sensitive(trust: dict, cfg: dict, primary: str) -> bool:
    if isinstance(cfg.get("level_sensitive"), bool):
        return cfg["level_sensitive"]
    engines = cfg.get("engines")
    if isinstance(engines, dict):
        eng = engines.get(primary)
        if isinstance(eng, dict) and isinstance(eng.get("level_sensitive"), bool):
            return eng["level_sensitive"]
    eng = trust.get("engine")
    if isinstance(eng, dict) and isinstance(eng.get("level_sensitive"), bool):
        return eng["level_sensitive"]
    return True


def _clip_fraction(trust: dict) -> float | None:
    metrics = trust.get("audio_metrics") if isinstance(trust.get("audio_metrics"), dict) else {}
    clip = metrics.get("clip_fraction")
    return float(clip) if _is_num(clip) else None


def _peak_dbfs(trust: dict) -> float | None:
    metrics = trust.get("audio_metrics") if isinstance(trust.get("audio_metrics"), dict) else {}
    peak = metrics.get("peak_dbfs")
    return float(peak) if _is_num(peak) else None


def _union(trust: dict, reasons: list[dict]) -> float | None:
    union = trust.get("coverage_union")
    if isinstance(union, dict) and _is_num(union.get("coverage_union")):
        return float(union["coverage_union"])
    if _is_num(union):
        return float(union)
    coverage = trust.get("coverage") if isinstance(trust.get("coverage"), dict) else {}
    if isinstance(coverage, dict):
        for row in coverage.values():
            if isinstance(row, dict) and _is_num(row.get("coverage_union")):
                return float(row["coverage_union"])
    for reason in reasons:
        if reason["code"] == "LOW_COVERAGE" and _is_num(reason.get("coverage_union")):
            return float(reason["coverage_union"])
    return None


def _primary_coverage(trust: dict, primary: str) -> dict:
    coverage = trust.get("coverage") if isinstance(trust.get("coverage"), dict) else {}
    row = coverage.get(primary)
    if isinstance(row, dict):
        return row
    for value in coverage.values():
        if isinstance(value, dict):
            return value
    return {}


def _merge_intervals(items: list[tuple[float, float]]) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in sorted(items, key=lambda pair: (pair[0], pair[1])):
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _valid_intervals(raw) -> list[tuple[float, float]]:
    out = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        start, end = item.get("start"), item.get("end")
        if _is_num(start) and _is_num(end) and float(end) > float(start):
            out.append((float(start), float(end)))
    return out


def _repair_intervals(trust: dict, reasons: list[dict], codes: list[str], primary: str, notes: list[str]) -> list[dict]:
    """Union holes for repair_decode: reason intervals -> uncovered_all -> coverage gaps."""
    raw: list[tuple[float, float]] = []
    for reason in reasons:
        if reason["code"] in REPAIR_CODES:
            raw.extend(_valid_intervals(reason.get("intervals")))
    if not raw:
        raw = _valid_intervals(trust.get("uncovered_all"))
    if not raw:
        cov = _primary_coverage(trust, primary)
        raw = _valid_intervals(cov.get("speech_gaps"))
        if raw:
            notes.append("repair_decode: интервалы взяты из coverage.speech_gaps — в reasons они не указаны")
    if "TAIL_UNCOVERED" in codes:
        cov = _primary_coverage(trust, primary)
        last = cov.get("last_segment_end")
        duration = _duration(trust)
        if _is_num(last) and duration - float(last) >= MIN_HOLE_S:
            raw.append((float(last), duration))
            notes.append(f"repair_decode: хвост {last}–{round(duration, 2)} с добавлен из coverage.last_segment_end")
    merged = _merge_intervals(raw)
    kept = [[round(start, 2), round(end, 2)] for start, end in merged if end - start >= MIN_HOLE_S - 1e-9]
    dropped = len(merged) - len(kept)
    if dropped:
        notes.append(f"repair_decode: отброшено дыр короче {MIN_HOLE_S} с — {dropped}")
    return [{"start": start, "end": end} for start, end in kept]


def _agreement_windows(trust: dict) -> tuple[list[dict], float | None]:
    agreement = trust.get("agreement") if isinstance(trust.get("agreement"), dict) else {}
    mean = agreement.get("mean_dice")
    mean = float(mean) if _is_num(mean) else None
    out = []
    for window in _as_list(agreement.get("windows"), "agreement.windows"):
        if not isinstance(window, dict) or not _is_num(window.get("dice")):
            continue
        dice = float(window["dice"])
        if dice >= DICE_BLOCK:
            continue
        start, end = window.get("start"), window.get("end")
        if not (_is_num(start) and _is_num(end)) or float(end) <= float(start):
            continue
        out.append({"start": float(start), "end": float(end), "dice": dice})
    return out, mean


def plan_route(trust: dict, cfg: dict | None = None) -> dict:
    """Map a trust card to an executable route plan (pure, no I/O)."""
    trust = _as_mapping(trust, "trust")
    cfg = _as_mapping(cfg, "cfg")

    reasons = _reasons(trust)
    codes: list[str] = []
    for reason in reasons:
        if reason["code"] not in codes:
            codes.append(reason["code"])
    notes: list[str] = []
    for reason in reasons:
        severity = reason.get("severity")
        if severity not in ("warn", "block"):
            notes.append(f"{reason['code']}: severity {severity!r} не распознан, считаем block (fail-closed)")
            reason["severity"] = "block"

    duration = _duration(trust)
    frac = cfg.get("budget_frac", DEFAULT_BUDGET_FRAC)
    if not _is_num(frac) or not 0 < float(frac) <= 1:
        raise RouteError("ROUTE_BAD_CFG", f"budget_frac должен быть в (0, 1], получено {frac!r}")
    budget = {"attempts_max": ATTEMPTS_MAX, "seconds_max": round(float(frac) * duration, 1)}
    if duration <= 0:
        notes.append("audio_metrics.duration_s отсутствует — секундный бюджет посчитан как 0")

    primary = _primary_name(trust, cfg)
    peer = _peer_name(trust, cfg, primary)
    level_sensitive = _level_sensitive(trust, cfg, primary)
    clip = _clip_fraction(trust)
    union = _union(trust, reasons)
    retry_done = bool(cfg.get("second_engine_done"))
    if not retry_done:
        attempts = trust.get("attempts")
        if isinstance(attempts, dict) and _is_num(attempts.get("second_engine")):
            retry_done = float(attempts["second_engine"]) > 0

    steps: dict[str, dict] = {}
    blockers: list[dict] = []

    if "LOW_LEVEL" in codes:
        peak = _peak_dbfs(trust)
        if not level_sensitive:
            notes.append(f"LOW_LEVEL: движок {primary} не level_sensitive — нормализация не планируется")
        elif peak is None:
            notes.append("LOW_LEVEL: нет audio_metrics.peak_dbfs — усиление не рассчитано")
        else:
            gain = min(TARGET_PEAK_DBFS - peak, GAIN_CAP_DB)
            if gain == GAIN_CAP_DB:
                notes.append(f"LOW_LEVEL: усиление ограничено {GAIN_CAP_DB:.0f} dB (нужно {round(TARGET_PEAK_DBFS - peak, 2)})")
            steps["normalize"] = {
                "id": "normalize",
                "kind": "gain",
                "because": "LOW_LEVEL",
                "params": {"gain_db": round(gain, 2), "target_peak_dbfs": TARGET_PEAK_DBFS, "peak_dbfs": peak},
            }

    if "VAD_UNRELIABLE" in codes:
        steps["detector"] = {
            "id": "detector",
            "kind": "vad",
            "because": "VAD_UNRELIABLE",
            "params": {"detector": "energy"},
        }

    if "CLIPPING" in codes:
        steps["no_gain"] = {
            "id": "no_gain",
            "kind": "gain_guard",
            "because": "CLIPPING",
            "params": {"gain_db": 0.0, "clip_fraction": clip if clip is not None else 0.0},
        }
        if clip is None:
            notes.append("CLIPPING: нет audio_metrics.clip_fraction — блокировка не включена")
        elif clip > CLIP_BLOCK_FRACTION:
            blockers.append(
                {
                    "code": "CLIPPING",
                    "reason_code": "CLIPPING",
                    "detail": f"клиппинг {clip:.4f} > {CLIP_BLOCK_FRACTION}: усиление не поможет, нужен источник",
                }
            )

    if "LOW_COVERAGE" in codes:
        if retry_done and union is not None and union < MIN_UNION_AFTER_REPEAT:
            blockers.append(
                {
                    "code": "LOW_COVERAGE",
                    "reason_code": "LOW_COVERAGE",
                    "detail": f"после повтора union={union} < {MIN_UNION_AFTER_REPEAT}",
                }
            )
        elif retry_done and union is None:
            notes.append("LOW_COVERAGE: повтор уже был, union неизвестен — второй движок запланирован снова")
            steps["second_engine"] = {
                "id": "second_engine",
                "kind": "engine",
                "because": "LOW_COVERAGE",
                "params": {"engine": peer, "min_union": MIN_UNION_AFTER_REPEAT},
            }
        elif retry_done:
            notes.append(f"LOW_COVERAGE: повтор уже был, union={union} >= {MIN_UNION_AFTER_REPEAT} — шаг не нужен")
        else:
            steps["second_engine"] = {
                "id": "second_engine",
                "kind": "engine",
                "because": "LOW_COVERAGE",
                "params": {"engine": peer, "min_union": MIN_UNION_AFTER_REPEAT},
            }

    if cfg.get("fill_from_peers") is True:
        steps["fill_from_peers"] = {
            "id": "fill_from_peers",
            "kind": "merge_peers",
            "because": "LOW_COVERAGE",
            "params": {"engines": [peer], "only_gaps": True},
        }

    repair_codes = [code for code in codes if code in REPAIR_CODES]
    if repair_codes:
        intervals = _repair_intervals(trust, reasons, codes, primary, notes)
        if intervals:
            steps["repair_decode"] = {
                "id": "repair_decode",
                "kind": "repair",
                "because": "SPEECH_GAP" if "SPEECH_GAP" in codes else "TAIL_UNCOVERED",
                "params": {"intervals": intervals},
            }
        else:
            notes.append("repair_decode: интервалы дыр не найдены — шаг не запланирован")

    if "LOW_AGREEMENT" in codes:
        windows, mean = _agreement_windows(trust)
        if windows:
            steps["arbitrate"] = {
                "id": "arbitrate",
                "kind": "arbitration",
                "because": "LOW_AGREEMENT",
                "params": {"max_dice": DICE_BLOCK, "windows": windows},
            }
        elif mean is not None and mean < DICE_BLOCK and duration > 0:
            steps["arbitrate"] = {
                "id": "arbitrate",
                "kind": "arbitration",
                "because": "LOW_AGREEMENT",
                "params": {
                    "max_dice": DICE_BLOCK,
                    "windows": [{"start": 0.0, "end": round(duration, 2), "dice": mean}],
                },
            }
            notes.append("LOW_AGREEMENT: окон нет — арбитраж на весь файл")
        else:
            notes.append("LOW_AGREEMENT: нет agreement.windows/mean_dice — арбитраж не спланирован")

    if "LOW_INFORMATION" in codes and "repair_decode" not in steps:
        reason = next((r for r in reasons if r.get("code") == "LOW_INFORMATION"), {})
        iv = reason.get("intervals") or []
        if iv:
            steps["repair_decode"] = {
                "id": "repair_decode",
                "kind": "repair",
                "because": "LOW_INFORMATION",
                "params": {"intervals": iv, "note": "вырожденные сегменты: пересегментировать и повторить"},
            }
        else:
            notes.append("LOW_INFORMATION: интервалы деградации не переданы — шаг не запланирован")

    for code in codes:
        if code not in KNOWN_CODES:
            blockers.append(
                {
                    "code": "UNKNOWN_REASON",
                    "reason_code": code,
                    "detail": f"неизвестный код причины {code!r}: маршрут не строится (fail-closed)",
                }
            )
    if "LANGUAGE_MISMATCH" in codes:
        blockers.append(
            {
                "code": "LANGUAGE_MISMATCH",
                "reason_code": "LANGUAGE_MISMATCH",
                "detail": "язык записи не совпал с ожидаемым: нужен иноязычный пайплайн",
            }
        )

    blockers.sort(key=lambda b: (BLOCK_PRIORITY.get(b["code"], 99), codes.index(b["reason_code"])))
    blocked = None
    if blockers:
        blocked = {
            "code": blockers[0]["code"],
            "codes": sorted({b["reason_code"] for b in blockers}, key=codes.index),
            "details": [b["detail"] for b in blockers],
            "action": "human_review",
        }

    block_severity = any(reason["severity"] == "block" for reason in reasons)
    derived = "failed" if (blockers or block_severity) else ("partial" if codes else "ok")
    trust_verdict = trust.get("verdict")
    verdict = derived
    if trust_verdict in VERDICT_RANK and VERDICT_RANK[trust_verdict] > VERDICT_RANK[derived]:
        verdict = trust_verdict
        notes.append(f"verdict: trust={trust_verdict} строже расчёта по severity — принят trust")
    elif trust_verdict in VERDICT_RANK and VERDICT_RANK[trust_verdict] < VERDICT_RANK[derived]:
        notes.append(f"verdict: trust={trust_verdict} мягче расчёта по severity — принят {derived}")

    summary_allowed = verdict != "failed" and not blockers
    summary_mode = {"ok": "full", "partial": "caveated", "failed": "blocked"}[verdict]

    if cfg.get("summarize") is True:
        steps["summarize"] = {
            "id": "summarize",
            "kind": "summary",
            "because": None,
            "params": {"mode": summary_mode},
        }

    if blockers:
        steps = {}  # fail-closed: автоматические шаги не выполняются
        notes.append(f"маршрут заблокирован: {blocked['code']} — нужен человек")

    addressed = {step["because"] for step in steps.values() if step.get("because")}
    unresolved = [code for code in codes if code not in addressed and code not in blocked_codes(blocked)]
    if unresolved:
        notes.append("не адресовано шагами: " + ", ".join(unresolved))

    ordered_steps = [steps[step_id] for step_id in STEP_ORDER if step_id in steps]
    ordered_steps = [_public_step(step) for step in ordered_steps]
    return {
        "schema": SCHEMA,
        "verdict": verdict,
        "codes": codes,
        "blocked": blocked,
        "steps": ordered_steps,
        "budget": budget,
        "summary_allowed": summary_allowed,
        "summary_mode": summary_mode,
        "unresolved": unresolved,
        "notes": notes,
    }


def blocked_codes(blocked: dict | None) -> set[str]:
    if not blocked:
        return set()
    return set(blocked.get("codes") or [])


def _public_step(step: dict) -> dict:
    """Keep the public step shape stable: id, kind, because, params."""
    return {
        "id": step["id"],
        "kind": step["kind"],
        "because": step.get("because"),
        "params": step.get("params") or {},
    }


def route_exit_code(route: dict) -> int:
    if route.get("blocked"):
        return EXIT_BLOCKED
    return EXIT_BY_VERDICT.get(route.get("verdict"), EXIT_INTERNAL)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # argparse exits 2 by default; contract wants 64
        raise RouteError("ROUTE_BAD_ARGS", message)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _Parser(description="Build an asr.route/1 plan from a trust card")
    parser.add_argument("--trust", required=True, help="path to trust.json")
    parser.add_argument("--out", required=True, help="route.json path or output directory")
    parser.add_argument("--budget-frac", type=float, default=DEFAULT_BUDGET_FRAC, help="repair budget share of duration")
    return parser.parse_args(argv)


def _out_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.suffix.lower() == ".json" or (path.exists() and path.is_file()):
        return path
    return path / "route.json"


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        trust_path = Path(args.trust).expanduser()
        if not trust_path.is_file():
            raise RouteError("ROUTE_BAD_ARGS", f"--trust: файл не найден: {trust_path}")
        try:
            trust = json.loads(trust_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RouteError("ROUTE_BAD_TRUST", f"{trust_path}: некорректный JSON ({exc})") from exc
        except OSError as exc:
            raise RouteError("ROUTE_BAD_ARGS", f"{trust_path}: {exc}") from exc
        if not isinstance(trust, dict):
            raise RouteError("ROUTE_BAD_TRUST", "верхний уровень trust.json должен быть объектом")
        route = plan_route(trust, {"budget_frac": args.budget_frac})
        out = _out_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(route, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(route, ensure_ascii=False, indent=1))
        return route_exit_code(route)
    except RouteError as exc:
        print(json.dumps({"error": exc.code, "detail": exc.detail}, ensure_ascii=False, indent=1))
        return EXIT_BAD_ARGS
    except Exception as exc:  # internal error: keep the CLI contract
        print(json.dumps({"error": "ROUTE_INTERNAL", "detail": str(exc)}, ensure_ascii=False, indent=1))
        return EXIT_INTERNAL


if __name__ == "__main__":
    sys.exit(main())
