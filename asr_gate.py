#!/usr/bin/env python3
"""asr_gate.py — контроль качества транскрипта (ASR gate), схема ``asr.trust/2``.

Что делает
  * аудио-аудит: длительность, RMS, пик, клиппинг, доля энергии ниже 1 кГц, RMS по 10 с;
  * активность: энергия (считается всегда — она знаменатель покрытия) + VAD Silero (опция);
  * LID: whisper / declared / off;
  * покрытие: ТОЛЬКО union интервалов сегментов, пересечённый с энергетической активностью;
  * дыры: непокрытые участки активности длиной >= MIN_GAP_S (пересечение с union, не вложенность);
  * согласие primary с каждым peer (dice по токенам на окнах AGREEMENT_WINDOW_S);
  * причины с severity (warn|block) и intervals -> вердикт ok|partial|failed.

Инварианты v2 (что исправлено относительно 1.x)
  1. Покрытие НЕ считается суммой длительностей сегментов: сумма давала 1.0 при дырах.
     covered_s = |union(segments) ∩ activity|, знаменатель = |activity| (энергия).
  2. Дыры ищутся пересечением активности с union, а не «вложенностью сегмента в интервал
     активности»: на реальном аудио вложенность давала 71.7 с ложных дыр против 20.9 с настоящих.
  3. --primary NAME фиксирует, чей транскрипт определяет вердикт; порядок --transcript
     на вердикт не влияет (без --primary берётся whisper, иначе лексикографически первый).
  4. --vad off даёт vad_active_s=None (а не 0.0), поэтому VAD_UNRELIABLE не выдумывается.
  5. .jsonl читается построчно (по одному объекту или сегменту на строку).
  6. Неверные аргументы -> exit 64 (JSON с ошибкой в stdout); пустой --transcript не падает трейсбеком.
  7. severity обязателен; block -> failed, warn -> partial; LOW_COVERAGE блокирует при union < 0.65.
  8. В trust.json пишутся schema=asr.trust/2, gate_version, coverage_union, uncovered_all, config.

CLI
  asr_gate.py --audio rec.wav --transcript whisper=whisper.json \\
              --transcript kts=kts.json --primary whisper \\
              --vad energy --lid declared --language ru --out ./gate

Exit-коды: 0 — ok, 2 — partial/failed (гейт не возвращает 3/4), 64 — неверные аргументы,
1 — внутренняя ошибка (битый файл и т.п.).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

GATE_VERSION = "2.0"
TRUST_SCHEMA = "asr.trust/2"

# --- пороги (все попадают в trust.json -> config.thresholds) -----------------
ENERGY_FRAME_S = 0.025
ENERGY_HOP_S = 0.01
ENERGY_THRESHOLD_DB = 8.0      # активность = RMS кадра выше шумового пола + 8 dB
ENERGY_FALLBACK_DBFS = -60.0   # если пауз нет вовсе: абсолютный пол, чтобы не получить 0 с активности
MERGE_GAP_S = 0.35             # склейка кадров активности в интервалы
MIN_GAP_S = 1.0                # дыра = непокрытая активность длиной >= 1.0 с
TAIL_ALERT_S = 10.0            # активность после последнего сегмента -> TAIL_UNCOVERED
LOW_LEVEL_DBFS = -45.0
CLIP_FRACTION = 0.001
VAD_UNDERREPORT_RATIO = 0.5    # VAD нашёл < 50% энергии -> VAD_UNRELIABLE
LOW_COVERAGE_WARN = 0.90
LOW_COVERAGE_BLOCK = 0.65
INFO_DEGENERATE_WARN = 0.30    # доля коротких сегментов без подтверждений -> LOW_INFORMATION
INFO_DEGENERATE_BLOCK = 0.50
INFO_MIN_SEGMENTS = 5
INFO_MIN_SEGMENT_S = 0.5
INFO_MIN_WORDS = 2
INFO_ACKNOWLEDGEMENTS = frozenset("да нет спасибо понял поняла хорошо ок ага угу верно точно".split())
LOW_AGREEMENT_DICE = 0.35
LOW_AGREEMENT_WINDOW_SHARE = 0.30
AGREEMENT_WINDOW_S = 20.0
LANG_MISMATCH_P = 0.8

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

THRESHOLDS = {
    "energy_threshold_db": ENERGY_THRESHOLD_DB,
    "energy_fallback_dbfs": ENERGY_FALLBACK_DBFS,
    "merge_gap_s": MERGE_GAP_S,
    "min_gap_s": MIN_GAP_S,
    "tail_alert_s": TAIL_ALERT_S,
    "low_level_dbfs": LOW_LEVEL_DBFS,
    "clip_fraction": CLIP_FRACTION,
    "vad_underreport_ratio": VAD_UNDERREPORT_RATIO,
    "low_coverage_warn": LOW_COVERAGE_WARN,
    "low_coverage_block": LOW_COVERAGE_BLOCK,
    "info_degenerate_warn": INFO_DEGENERATE_WARN,
    "info_degenerate_block": INFO_DEGENERATE_BLOCK,
    "info_min_segments": INFO_MIN_SEGMENTS,
    "info_min_segment_s": INFO_MIN_SEGMENT_S,
    "info_min_words": INFO_MIN_WORDS,
    "low_agreement_dice": LOW_AGREEMENT_DICE,
    "low_agreement_window_share": LOW_AGREEMENT_WINDOW_SHARE,
    "agreement_window_s": AGREEMENT_WINDOW_S,
    "language_mismatch_p": LANG_MISMATCH_P,
    "lid_offline": True,
}

RU_STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть
надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто
этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее кажется сейчас
были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через
эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше
чуть том нельзя такой им более всегда конечно всю между""".split())


class UsageError(Exception):
    """Неверные аргументы CLI -> exit 64."""


class _Parser(argparse.ArgumentParser):
    """argparse, который не печатает usage и не делает exit(2), а кидает UsageError."""

    def error(self, message):  # noqa: D401 - контракт argparse
        raise UsageError(f"BAD_ARGS: {message}")


# ---------------------------------------------------------------------------
# интервальная арифметика
# ---------------------------------------------------------------------------
Span = tuple[float, float]


def merge_intervals(spans, gap: float = 0.0) -> list[Span]:
    """Объединение интервалов. gap — допустимый разрыв для склейки (0 = строгий union)."""
    vals = sorted((float(a), float(b)) for a, b in spans if float(b) > float(a))
    out: list[list[float]] = []
    for a, b in vals:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def union_length(spans) -> float:
    return float(sum(b - a for a, b in merge_intervals(spans)))


def intersect(a, b) -> list[Span]:
    """Пересечение двух наборов интервалов (оба нормализуются)."""
    out: list[Span] = []
    for alo, ahi in merge_intervals(a):
        for blo, bhi in merge_intervals(b):
            lo, hi = max(alo, blo), min(ahi, bhi)
            if hi > lo:
                out.append((lo, hi))
    return merge_intervals(out)


def subtract(a, b) -> list[Span]:
    """a минус b (оба нормализуются)."""
    out: list[Span] = []
    bs = merge_intervals(b)
    for start, end in merge_intervals(a):
        pieces: list[Span] = [(start, end)]
        for blo, bhi in bs:
            nxt: list[Span] = []
            for lo, hi in pieces:
                if bhi <= lo or blo >= hi:
                    nxt.append((lo, hi))
                    continue
                if lo < blo:
                    nxt.append((lo, min(blo, hi)))
                if bhi < hi:
                    nxt.append((max(bhi, lo), hi))
            pieces = nxt
        out.extend(pieces)
    return merge_intervals(out)


def to_interval_dicts(spans) -> list[dict]:
    rows = list(spans or [])
    if rows and isinstance(rows[0], dict):
        spans = from_interval_dicts(rows)
    else:
        spans = rows
    return [{"start": round(a, 2), "end": round(b, 2)} for a, b in merge_intervals(spans)]


def from_interval_dicts(rows) -> list[Span]:
    out: list[Span] = []
    for r in rows or []:
        if isinstance(r, dict) and r.get("start") is not None and r.get("end") is not None:
            out.append((float(r["start"]), float(r["end"])))
        elif isinstance(r, (list, tuple)) and len(r) == 2:
            out.append((float(r[0]), float(r[1])))
    return merge_intervals(out)


def _merge(frames, gap: float = MERGE_GAP_S) -> list[dict]:
    """Совместимость с 1.x: склейка кадров и отдача словарями."""
    return to_interval_dicts(merge_intervals(frames, gap))


# ---------------------------------------------------------------------------
# аргументы и валидация
# ---------------------------------------------------------------------------
@dataclass
class GateArgs:
    audio: Path
    transcripts: dict[str, Path]
    evidence: dict[str, Path]
    primary: str
    vad: str
    lid: str
    language: str
    out: Path
    stt_model: str


def parse_args(argv=None) -> argparse.Namespace:
    p = _Parser(
        prog="asr_gate.py",
        description="ASR quality gate: аудио-аудит + активность + LID + покрытие -> trust.json",
    )
    p.add_argument("--audio", required=True, metavar="PATH")
    p.add_argument("--transcript", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--evidence", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--primary", default=None, metavar="NAME",
                   help="чей транскрипт определяет вердикт (по умолчанию whisper, иначе первый по алфавиту)")
    p.add_argument("--vad", choices=["silero", "energy", "off"], default="silero")
    p.add_argument("--lid", choices=["whisper", "declared", "off"], default="whisper")
    p.add_argument("--language", default="ru")
    p.add_argument("--out", required=True, metavar="DIR")
    p.add_argument("--stt-model", default="large-v3")
    return p.parse_args(argv)


def _abs_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _parse_spec(spec: str, flag: str) -> tuple[str, Path]:
    name, sep, raw = spec.partition("=")
    name, raw = name.strip(), raw.strip()
    if not sep:
        raise UsageError(f"{flag}: ожидалось NAME=PATH, получено {spec!r}")
    if not name or not NAME_RE.match(name):
        raise UsageError(f"{flag}: недопустимое имя {name!r} (буквы/цифры/._-)")
    if not raw:
        raise UsageError(f"{flag} {name}: пустой путь")
    path = _abs_path(raw)
    if not path.is_file():
        raise UsageError(f"{flag} {name}: файл не найден: {path}")
    return name, path


def validate_args(a: argparse.Namespace) -> GateArgs:
    audio = _abs_path(a.audio)
    if not audio.is_file():
        raise UsageError(f"--audio: файл не найден: {audio}")
    if not a.transcript:
        raise UsageError("--transcript обязателен: нужен хотя бы один NAME=PATH")

    transcripts: dict[str, Path] = {}
    for spec in a.transcript:
        name, path = _parse_spec(spec, "--transcript")
        if name in transcripts:
            raise UsageError(f"--transcript: имя {name!r} задано дважды")
        transcripts[name] = path

    evidence: dict[str, Path] = {}
    for spec in a.evidence:
        name, path = _parse_spec(spec, "--evidence")
        evidence[name] = path

    if a.primary is None:
        primary = "whisper" if "whisper" in transcripts else sorted(transcripts)[0]
    else:
        primary = a.primary.strip()
        if primary not in transcripts:
            raise UsageError(
                f"--primary {primary!r} отсутствует среди --transcript: {sorted(transcripts)}"
            )

    out = _abs_path(a.out)
    if out.exists() and not out.is_dir():
        raise UsageError(f"--out: это файл, а нужен каталог: {out}")
    return GateArgs(audio=audio, transcripts=transcripts, evidence=evidence, primary=primary,
                    vad=a.vad, lid=a.lid, language=a.language, out=out, stt_model=a.stt_model)


# ---------------------------------------------------------------------------
# загрузка транскриптов
# ---------------------------------------------------------------------------
def _rows_of(data) -> list:
    if isinstance(data, dict):
        rows = data.get("segments")
        if rows is None:
            rows = data.get("windows")
        if rows is None:
            rows = data.get("rows")
        return list(rows or [])
    if isinstance(data, list):
        return data
    return []


def _rows_from_jsonl(text: str, path: Path) -> list:
    rows: list = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"LOAD_FAILED: {path}:{i}: битая строка JSONL: {e}") from e
        if isinstance(obj, dict) and ("segments" in obj or "windows" in obj):
            rows.extend(_rows_of(obj))
        elif isinstance(obj, list):
            rows.extend(obj)
        else:
            rows.append(obj)
    return rows


def _parse_text(text: str, path: Path) -> list:
    """Один JSON-документ или JSONL: сначала целиком, при неудаче — построчно."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _rows_from_jsonl(text, path)
    if isinstance(data, dict) and not ("segments" in data or "windows" in data or "rows" in data):
        # одиночный сегмент/объект без обёртки — тоже читаем построчно
        return _rows_from_jsonl(text, path)
    return _rows_of(data)


def load_segments(path) -> list[dict]:
    """Читает транскрипт (.json / .jsonl) -> [{'start','end','text'}].

    .jsonl читается построчно (один объект или один сегмент на строку); .json — целиком,
    но если это на самом деле JSONL, файл дочитывается построчно.
    """
    path = Path(path)
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError(f"LOAD_FAILED: неподдерживаемый формат транскрипта: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"LOAD_FAILED: {path}: {e}") from e

    rows = _parse_text(text, path)

    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        start, end = r.get("start"), r.get("end")
        if start is None or end is None:
            bounds = r.get("b") or r.get("bounds") or []
            if len(bounds) == 2:
                start = bounds[0] if start is None else start
                end = bounds[1] if end is None else end
        if start is None or end is None:
            continue
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        text_val = r.get("text")
        out.append({"start": start, "end": end, "text": "" if text_val is None else str(text_val)})
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


# ---------------------------------------------------------------------------
# аудио-метрики и активность
# ---------------------------------------------------------------------------
def audio_metrics(x: np.ndarray, sr: int) -> dict:
    dur = len(x) / sr
    if len(x) == 0:
        raise ValueError("AUDIO_EMPTY: пустой сигнал")
    rms = float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))
    peak = float(20 * np.log10(np.max(np.abs(x)) + 1e-12))
    clip = float(np.mean(np.abs(x) >= 0.999))
    head = x[: sr * min(60, int(dur))] if dur >= 1 else x
    spec, freqs = np.abs(np.fft.rfft(head)), np.fft.rfftfreq(len(head), 1 / sr)
    low = float(spec[freqs < 1000].sum() / (spec.sum() + 1e-12))
    step = 10 * sr
    per10 = [round(float(20 * np.log10(np.sqrt(np.mean(x[i:i + step].astype(np.float64) ** 2)) + 1e-12)), 1)
             for i in range(0, len(x), step) if len(x[i:i + step])]
    return {"duration_s": round(dur, 2), "rms_dbfs": round(rms, 2), "peak_dbfs": round(peak, 2),
            "clip_fraction": round(clip, 6), "band_share_below_1khz": round(low, 3),
            "rms_per_10s": per10}


def energy_activity(x: np.ndarray, sr: int) -> dict:
    """Активность по энергии: RMS кадра > шумовой пол + ENERGY_THRESHOLD_DB."""
    frame, hop = int(ENERGY_FRAME_S * sr), int(ENERGY_HOP_S * sr)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    starts = range(0, len(x) - frame + 1, hop)
    levels = np.array([20 * np.log10(np.sqrt(np.mean(x[i:i + frame].astype(np.float64) ** 2)) + 1e-12)
                       for i in starts])
    floor = float(np.percentile(levels, 10)) if len(levels) else -120.0
    active = levels > floor + ENERGY_THRESHOLD_DB
    rms = float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))
    if not active.any() and rms > ENERGY_FALLBACK_DBFS:
        # пауз нет вовсе (перцентильный пол сидит внутри речи) — абсолютный порог,
        # иначе знаменатель покрытия обнулится и гейт молча пропустит файл
        active = levels > ENERGY_FALLBACK_DBFS
    frames = [(i * hop / sr, (i * hop + frame) / sr) for i, on in enumerate(active) if on]
    iv = merge_intervals(frames, MERGE_GAP_S)
    return {"active_s": round(union_length(iv), 1), "noise_floor_dbfs": round(floor, 1),
            "intervals": to_interval_dicts(iv)}


def vad_activity(x: np.ndarray, sr: int) -> dict:
    """Silero VAD (ML, требует faster-whisper)."""
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    # Silero амплитудно-чувствителен: кормим нормализованную копию, не тихий оригинал
    peak = float(np.max(np.abs(x))) or 1.0
    y = x * (0.9 / peak)
    ts = get_speech_timestamps(y, VadOptions(min_silence_duration_ms=500, speech_pad_ms=100))
    iv = merge_intervals([(t["start"] / sr, t["end"] / sr) for t in ts], MERGE_GAP_S)
    return {"active_s": round(union_length(iv), 1), "intervals": to_interval_dicts(iv)}


def _activity_for(mode: str, x: np.ndarray, sr: int, energy: dict) -> dict:
    """Возвращает {'active_s': float|None, 'intervals': [...], 'provider': str}."""
    if mode == "off":
        # None, а не 0.0: иначе 0.0 < 0.5 * energy даёт ложный VAD_UNRELIABLE
        return {"active_s": None, "intervals": [], "provider": "off"}
    if mode == "energy":
        return {"active_s": energy["active_s"], "intervals": energy["intervals"], "provider": "energy"}
    try:
        vad = vad_activity(x, sr)
        return {"active_s": vad["active_s"], "intervals": vad["intervals"], "provider": "silero"}
    except Exception as e:  # нет модели/onnxruntime/сети — не роняем гейт
        print(f"asr_gate: VAD_UNRELIABLE_DETECTOR: silero недоступен ({type(e).__name__}: {e})",
              file=sys.stderr)
        return {"active_s": None, "intervals": [], "provider": "silero_unavailable"}


def detect_language(x: np.ndarray, sr: int, model_name: str) -> dict:
    from faster_whisper import WhisperModel

    win = 30 * sr
    if len(x) > win:
        starts = range(0, len(x) - win, sr)
        i = max(starts, key=lambda s: float(np.mean(x[s:s + win].astype(np.float64) ** 2)))
        sample = x[i:i + win]
    else:
        sample = x
    m = WhisperModel(model_name, device="cpu", compute_type="int8", local_files_only=True)
    _, info = m.transcribe(sample, language=None, beam_size=1, vad_filter=False, temperature=0.0)
    return {"language": info.language, "probability": round(float(info.language_probability), 3),
            "method": "whisper"}


def _language_for(mode: str, x: np.ndarray, sr: int, language: str, model_name: str) -> dict:
    if mode == "off":
        return {"language": language, "probability": None, "method": "off"}
    if mode == "declared":
        return {"language": language, "probability": 1.0, "method": "declared"}
    try:
        return detect_language(x, sr, model_name)
    except Exception as e:  # модель не скачана и т.п.
        print(f"asr_gate: LID_UNAVAILABLE: whisper LID недоступен ({type(e).__name__}: {e})",
              file=sys.stderr)
        return {"language": None, "probability": None, "method": "whisper_unavailable"}


# ---------------------------------------------------------------------------
# покрытие и дыры
# ---------------------------------------------------------------------------
def coverage(segments: list[dict], duration: float, activity) -> dict:
    """Покрытие активности union'ом сегментов (НЕ суммой длительностей).

    covered_s       — секунды энергетической активности, покрытые union'ом сегментов;
    union_active_s  — вся энергетическая активность (знаменатель);
    coverage_union  — covered_s / union_active_s (clamp 1.0);
    coverage        — алиас coverage_union (совместимость с 1.x, где читали .coverage);
    tail_uncovered_s— активность после конца последнего сегмента;
    speech_gaps     — непокрытые участки активности длиной >= MIN_GAP_S;
    empty_segments  — число сегментов без текста (включая пробелы), без изменения union.
    """
    raw = activity.get("intervals", []) if isinstance(activity, dict) else activity
    act = merge_intervals(from_interval_dicts(raw))
    act = merge_intervals((max(0.0, a), min(duration, b)) for a, b in act)
    active_s = union_length(act)

    seg = merge_intervals((max(0.0, float(s["start"])), min(duration, float(s["end"])))
                          for s in segments if float(s["end"]) > 0.0 and float(s["start"]) < duration)
    covered = intersect(seg, act)
    covered_s = union_length(covered)
    cov = min(covered_s / active_s, 1.0) if active_s > 0 else None

    last_end = max((b for _, b in seg), default=0.0)
    tail = union_length(subtract(act, [(0.0, last_end)])) if last_end else active_s

    gaps = [(a, b) for a, b in subtract(act, seg) if b - a >= MIN_GAP_S - 1e-9]
    return {
        "coverage": round(cov, 3) if cov is not None else None,
        "coverage_union": round(cov, 3) if cov is not None else None,
        "covered_s": round(covered_s, 1),
        "union_active_s": round(active_s, 1),
        "tail_uncovered_s": round(tail, 1),
        "speech_gaps": to_interval_dicts(gaps),
        "last_segment_end": round(last_end, 2),
        "n_segments": len(segments),
        "empty_segments": sum(not str(s.get("text") or "").strip() for s in segments),
    }


def coverage_for(segments: list[dict], duration: float, activity) -> dict:
    return coverage(segments, duration, activity)


# ---------------------------------------------------------------------------
# согласие движков и словарь
# ---------------------------------------------------------------------------
def tokens(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def agreement(a: list[dict], b: list[dict], window: float, dur: float) -> dict:
    out = []
    for i in range(int(np.ceil(dur / window))):
        lo, hi = i * window, (i + 1) * window
        ta = set(tokens(" ".join(s["text"] for s in a if s["end"] > lo and s["start"] < hi)))
        tb = set(tokens(" ".join(s["text"] for s in b if s["end"] > lo and s["start"] < hi)))
        if not ta and not tb:
            continue
        dice = 2 * len(ta & tb) / (len(ta) + len(tb)) if (ta or tb) else 0.0
        out.append({"start": int(lo), "end": int(hi), "dice": round(dice, 2), "dice_raw": dice})
    vals = [r["dice_raw"] for r in out]
    mean = float(np.mean(vals)) if vals else None
    return {"mean_dice": round(mean, 2) if mean is not None else None,
            "mean_dice_raw": mean,
            "low_window_share": sum(v < LOW_AGREEMENT_DICE for v in vals) / len(vals) if vals else None,
            "windows": out}


def vocab(transcripts: dict[str, list[dict]], evidence: dict[str, str]) -> list[dict]:
    seen: dict[str, set[str]] = {}
    for name, segs in transcripts.items():
        for w in tokens(" ".join(s["text"] for s in segs)):
            if len(w) >= 8 and w not in RU_STOP:
                seen.setdefault(w, set()).add(name)
    for name, text in evidence.items():
        for w in tokens(text):
            if len(w) >= 8 and w not in RU_STOP:
                seen.setdefault(w, set()).add(name)
    rows = [{"term": w, "sources": sorted(src), "count": len(src)} for w, src in seen.items() if len(src) >= 2]
    return sorted(rows, key=lambda r: (-r["count"], r["term"]))[:40]


# ---------------------------------------------------------------------------
# причины и вердикт
# ---------------------------------------------------------------------------
def verdict_of(reasons: list[dict]) -> str:
    sev = {r["severity"] for r in reasons}
    if "block" in sev:
        return "failed"
    return "partial" if reasons else "ok"


def verdict_legacy_of(reasons: list[dict]) -> str:
    """Поведение 1.x: failed только на LANGUAGE_MISMATCH/LOW_COVERAGE."""
    codes = {r["code"] for r in reasons}
    if "LANGUAGE_MISMATCH" in codes or "LOW_COVERAGE" in codes:
        return "failed"
    return "partial" if reasons else "ok"


def build_reasons(*, metrics: dict, activity: dict, coverage: dict, lid: dict, language: str,
                  agreement: dict | None = None, energy_intervals=None,
                  segments: list[dict] | None = None) -> list[dict]:
    reasons: list[dict] = []

    def add(code: str, severity: str, detail: str, intervals=None):
        reasons.append({"code": code, "severity": severity, "detail": detail,
                        "intervals": to_interval_dicts(intervals or [])})

    prob = lid.get("probability")
    if (lid.get("method") not in {"off", "whisper_unavailable"}
            and lid.get("language") and lid["language"] != language
            and float(prob or 0.0) >= LANG_MISMATCH_P):
        add("LANGUAGE_MISMATCH", "block",
            f"LID={lid['language']} p={prob}, ожидался {language}")

    if metrics["rms_dbfs"] < LOW_LEVEL_DBFS:
        add("LOW_LEVEL", "warn", f"RMS {metrics['rms_dbfs']} dBFS < {LOW_LEVEL_DBFS} dBFS")
    if metrics["clip_fraction"] > CLIP_FRACTION:
        add("CLIPPING", "warn", f"{metrics['clip_fraction'] * 100:.2f}% отсчётов на пределе")

    vad_s = activity.get("vad_active_s")
    energy_s = float(activity.get("energy_active_s") or 0.0)
    if vad_s is not None and energy_s > 0 and float(vad_s) < VAD_UNDERREPORT_RATIO * energy_s:
        add("VAD_UNRELIABLE", "warn",
            f"VAD нашёл {vad_s} с речи против {energy_s} с активности по энергии")

    tail = float(coverage.get("tail_uncovered_s") or 0.0)
    if tail > TAIL_ALERT_S:
        tail_iv = subtract(from_interval_dicts(energy_intervals or []),
                           [(0.0, float(coverage.get("last_segment_end") or 0.0))])
        add("TAIL_UNCOVERED", "warn",
            f"{tail} с активности после последнего сегмента "
            f"(конец {coverage.get('last_segment_end')} с)", tail_iv)

    cov_u = coverage.get("coverage_union")
    if cov_u is not None and cov_u < LOW_COVERAGE_WARN:
        block = cov_u < LOW_COVERAGE_BLOCK
        add("LOW_COVERAGE", "block" if block else "warn",
            f"union-покрытие активности {cov_u} < {LOW_COVERAGE_WARN}"
            + (f" (block при < {LOW_COVERAGE_BLOCK})" if block else ""))

    gaps = coverage.get("speech_gaps") or []
    if gaps:
        total = sum(g["end"] - g["start"] for g in gaps)
        longest = max(g["end"] - g["start"] for g in gaps)
        add("SPEECH_GAP", "warn",
            f"{len(gaps)} непокрытых участков активности >= {MIN_GAP_S} с "
            f"(всего {round(total, 1)} с, макс {round(longest, 1)} с)", gaps)

    if agreement:
        scores = {name: dice for name, dice in
                  (agreement.get("per_peer_raw", agreement.get("per_peer")) or {}).items()
                  if dice is not None}
        worst_peer = min(scores, key=scores.get) if scores else None
        worst_dice = (scores[worst_peer] if worst_peer is not None else
                      agreement.get("mean_dice_raw", agreement.get("mean_dice")))
        groups = agreement.get("windows_per_peer")
        if groups is None:
            groups = {"": agreement.get("windows", [])}
        shares = {name: sum(w.get("dice_raw", w["dice"]) < LOW_AGREEMENT_DICE for w in rows) / len(rows)
                  for name, rows in groups.items() if rows}
        worst_share_peer = max(shares, key=shares.get) if shares else None
        worst_share = (shares[worst_share_peer] if worst_share_peer is not None else
                       agreement.get("low_window_share"))
        details = []
        if worst_dice is not None and worst_dice < LOW_AGREEMENT_DICE:
            peer_detail = f" с «{worst_peer}»" if worst_peer is not None else ""
            details.append(f"согласие{peer_detail} {worst_dice} < {LOW_AGREEMENT_DICE}")
        if worst_share is not None and worst_share >= LOW_AGREEMENT_WINDOW_SHARE:
            peer_detail = f" с «{worst_share_peer}»" if worst_share_peer else ""
            details.append(f"доля окон с Dice < {LOW_AGREEMENT_DICE}{peer_detail}: "
                           f"{worst_share} >= {LOW_AGREEMENT_WINDOW_SHARE}")
        if details:
            low = [w for rows in groups.values() for w in rows
                   if w.get("dice_raw", w["dice"]) < LOW_AGREEMENT_DICE]
            add("LOW_AGREEMENT", "warn", "; ".join(details),
                [(w["start"], w["end"]) for w in low])

    considered = [s for s in (segments or [])
                  if float(s.get("end") or 0) - float(s.get("start") or 0) >= INFO_MIN_SEGMENT_S]
    empty = [s for s in (segments or []) if not tokens(str(s.get("text") or ""))]
    # Пустой текст блокирует независимо от длительности и числа сегментов.
    # None означает, что вызывающий код не передал текст для проверки; [] — пустой транскрипт.
    if segments is not None and len(empty) == len(segments):
        add("LOW_INFORMATION", "block",
            f"транскрипт без текста ({len(segments)} сегментов): интервальное покрытие не подтверждает текст",
            [(float(s["start"]), float(s["end"])) for s in segments[:20]])
    else:
        degenerate = [s for s in considered if len(tokens(str(s.get("text") or ""))) < INFO_MIN_WORDS]
        has_confirmation = any(w in INFO_ACKNOWLEDGEMENTS for s in degenerate
                               for w in tokens(str(s.get("text") or "")))
        frac = (len(degenerate) / len(considered)
                if len(considered) >= INFO_MIN_SEGMENTS and not has_confirmation else 0.0)
        total = len(considered)
        description = f"сегментов короче {INFO_MIN_WORDS} слов без реплик-подтверждений"
        # Подтверждения оправдывают короткие реплики, но не массовую потерю текста.
        # Пустые считаем среди ВСЕХ сегментов, в том числе короче INFO_MIN_SEGMENT_S.
        empty_frac = len(empty) / len(segments) if segments else 0.0
        if len(empty) >= 2 and empty_frac > INFO_DEGENERATE_WARN and empty_frac > frac:
            degenerate, frac, total = empty, empty_frac, len(segments)
            description = "сегментов без текста"
        if frac >= INFO_DEGENERATE_WARN:
            sev = "block" if frac >= INFO_DEGENERATE_BLOCK else "warn"
            add("LOW_INFORMATION", sev,
                f"{len(degenerate)} из {total} {description} ({frac:.2f}): возможна деградация ASR",
                [(float(s["start"]), float(s["end"])) for s in degenerate[:20]])

    return reasons


def actions_for(reasons: list[dict]) -> str:
    route = {
        "LANGUAGE_MISMATCH": "иноязычный пайплайн (Whisper), не считать текст русским",
        "LOW_LEVEL": "нормализовать до пика -1 dBFS и повторить",
        "CLIPPING": "проверить усиление: клиппинг в источнике",
        "VAD_UNRELIABLE": "не верить VAD: сегментировать по энергии или окнами",
        "TAIL_UNCOVERED": "дописать хвост без VAD",
        "LOW_COVERAGE": "второй движок и сегментация без VAD",
        "SPEECH_GAP": "точечный повтор пропущенных участков",
        "LOW_AGREEMENT": "прослушать спорные участки, второе мнение обязательно",
        "LOW_INFORMATION": "пересегментировать по энергии и повторить: однословные сегменты — деградация ASR",
    }
    codes = {r["code"] for r in reasons}
    acts = [route[c] for c in route if c in codes]
    return "; ".join(acts) if acts else "текст можно отдавать"


# ---------------------------------------------------------------------------
# отчёты
# ---------------------------------------------------------------------------
def _trust_md(report: dict, vocab_rows: list[dict]) -> str:
    m, act, lid = report["audio_metrics"], report["activity"], report["language_id"]
    lines = [f"# Карточка доверия — {Path(report['audio']).name}", "",
             f"**Вердикт: `{report['verdict']}`** (primary: `{report['primary']}`, "
             f"gate {report['gate_version']})", "",
             f"Действие: {report['action']}", ""]
    if report["reasons"]:
        lines += ["## Почему", ""]
        for r in report["reasons"]:
            iv = ""
            if r["intervals"]:
                iv = " — " + ", ".join(f"{i['start']}–{i['end']} с" for i in r["intervals"][:5])
                if len(r["intervals"]) > 5:
                    iv += f" (+{len(r['intervals']) - 5})"
            lines.append(f"- `{r['code']}` [{r['severity']}] — {r['detail']}{iv}")
        lines.append("")
    lines += ["## Аудио", "",
              f"- длительность {m['duration_s']} с, RMS {m['rms_dbfs']} dBFS, пик {m['peak_dbfs']} dBFS",
              f"- клиппинг {m['clip_fraction'] * 100:.3f} %, энергия ниже 1 кГц {m['band_share_below_1khz']}",
              f"- активность: энергия {act['energy_active_s']} с, VAD ({act.get('vad')}) "
              f"{act['vad_active_s']} с; шумовой пол {act['noise_floor_dbfs']} dB",
              f"- язык: {lid['language']} (p={lid['probability']}, {lid['method']})", ""]
    cu = report["coverage_union"]
    lines += ["## Покрытие (union активности)", "",
              f"Основной транскрипт: covered {cu['covered_s']} с из {cu['active_s']} с "
              f"= **{cu['coverage_union']}**", "",
              "| транскрипт | union-покрытие | покрыто, с | активно, с | хвост без текста | дыр |",
              "|---|---:|---:|---:|---:|---:|"]
    for n, c in report["coverage"].items():
        lines.append(f"| {n} | {c['coverage_union']} | {c['covered_s']} | {c['union_active_s']} | "
                     f"{c['tail_uncovered_s']} с | {len(c['speech_gaps'])} |")
    if report["uncovered_all"]:
        lines += ["", "Непокрытые участки активности: "
                  + ", ".join(f"{g['start']}–{g['end']}" for g in report["uncovered_all"][:12])
                  + (f" (+{len(report['uncovered_all']) - 12})" if len(report["uncovered_all"]) > 12 else "")]
    if report.get("agreement"):
        agr = report["agreement"]
        if agr.get("per_peer"):
            lines += ["", f"Среднее согласие «{agr['pair'][0]}» со всеми peers: {agr['mean_dice']}", ""]
            lines += [f"- «{name}»: {dice}" for name, dice in agr["per_peer"].items()]
        else:
            lines += ["", f"Согласие «{agr['pair'][0]}» / «{agr['pair'][1]}»: {agr['mean_dice']}"]
    lines += ["", "## Кандидаты в словарь", ""]
    lines += [f"- {r['term']} ({', '.join(r['sources'])})" for r in vocab_rows] or ["—"]
    return "\n".join(lines) + "\n"


def _write_outputs(report: dict, out: Path, vocab_rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "trust.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
    (out / "trust.md").write_text(_trust_md(report, vocab_rows), encoding="utf-8")
    (out / "vocab.md").write_text(
        "# Кандидаты в словарь\n\n"
        + ("\n".join(f"- {r['term']} — {', '.join(r['sources'])}" for r in vocab_rows) or "—")
        + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# запуск
# ---------------------------------------------------------------------------
def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1).astype(np.float32)
    if len(x) == 0:
        raise ValueError(f"AUDIO_EMPTY: {path}")
    return x, int(sr)


def run_gate(ctx: GateArgs) -> dict:
    x, sr = _read_audio(ctx.audio)
    dur = len(x) / sr

    metrics = audio_metrics(x, sr)
    energy = energy_activity(x, sr)
    vad = _activity_for(ctx.vad, x, sr, energy)
    lid = _language_for(ctx.lid, x, sr, ctx.language, ctx.stt_model)

    transcripts = {name: load_segments(path) for name, path in ctx.transcripts.items()}
    evidence = {name: path.read_text(encoding="utf-8") for name, path in ctx.evidence.items()}

    cov = {name: coverage(segs, dur, energy) for name, segs in transcripts.items()}

    agr = None
    others = sorted(n for n in transcripts if n != ctx.primary)
    if others:
        pairs = {name: agreement(transcripts[ctx.primary], transcripts[name], AGREEMENT_WINDOW_S, dur)
                 for name in others}
        per_peer = {name: pair["mean_dice"] for name, pair in pairs.items()}
        per_peer_raw = {name: pair["mean_dice_raw"] for name, pair in pairs.items()}
        scores = [dice for dice in per_peer_raw.values() if dice is not None]
        mean = float(np.mean(scores)) if scores else None
        shares = {name: pair["low_window_share"] for name, pair in pairs.items()}
        agr = {
            "mean_dice": round(mean, 2) if mean is not None else None,
            "mean_dice_raw": mean,
            "per_peer": per_peer,
            "per_peer_raw": per_peer_raw,
            # Как и среднее, долю проверяем для худшего peer: хорошие не скрывают плохого.
            "low_window_share": max((s for s in shares.values() if s is not None), default=None),
            "low_window_share_per_peer": shares,
            # Старые поля по-прежнему относятся к первому peer (по алфавиту).
            "windows": pairs[others[0]]["windows"],
            "pair": [ctx.primary, others[0]],
            "windows_per_peer": {name: pair["windows"] for name, pair in pairs.items()},
        }

    activity = {"energy_active_s": energy["active_s"], "vad_active_s": vad["active_s"],
                "noise_floor_dbfs": energy["noise_floor_dbfs"], "vad": vad["provider"]}

    reasons = build_reasons(metrics=metrics, activity=activity, coverage=cov[ctx.primary],
                            lid=lid, language=ctx.language, agreement=agr,
                            energy_intervals=energy["intervals"],
                            segments=transcripts[ctx.primary])
    verdict = verdict_of(reasons)
    cov_primary = cov[ctx.primary]

    report = {
        "schema": TRUST_SCHEMA,
        "verdict": verdict,
        "primary": ctx.primary,
        "gate_version": GATE_VERSION,
        "audio": str(ctx.audio),
        "action": actions_for(reasons),
        "reasons": reasons,
        "audio_metrics": metrics,
        "activity": activity,
        "language_id": lid,
        "coverage": cov,
        "empty_segments_total": sum(c["empty_segments"] for c in cov.values()),
        "coverage_union": {
            "covered_s": cov_primary["covered_s"],
            "active_s": cov_primary["union_active_s"],
            "coverage_union": cov_primary["coverage_union"],
        },
        "uncovered_all": cov_primary["speech_gaps"],
        "agreement": agr,
        "config": {"vad": ctx.vad, "lid": ctx.lid, "language": ctx.language,
                   "primary": ctx.primary, "thresholds": dict(THRESHOLDS)},
        "verdict_legacy": verdict_legacy_of(reasons),
    }
    _write_outputs(report, ctx.out, vocab(transcripts, evidence))
    return report


def main(argv=None) -> int:
    try:
        ctx = validate_args(parse_args(argv))
    except UsageError as e:
        print(json.dumps({"error": str(e), "exit": 64}, ensure_ascii=False))
        return 64

    try:
        report = run_gate(ctx)
    except Exception as e:  # битый аудиофайл/транскрипт — внутренняя ошибка, не 64
        print(json.dumps({"error": f"{type(e).__name__}: {e}", "exit": 1}, ensure_ascii=False))
        return 1

    summary = {
        "schema": report["schema"],
        "verdict": report["verdict"],
        "primary": report["primary"],
        "gate_version": report["gate_version"],
        "reasons": [{"code": r["code"], "severity": r["severity"]} for r in report["reasons"]],
        "coverage_union": report["coverage_union"],
        "uncovered_all": report["uncovered_all"],
        "activity": report["activity"],
        "language_id": report["language_id"],
        "agreement": report["agreement"]["mean_dice"] if report["agreement"] else None,
        "out": str(ctx.out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if report["verdict"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
