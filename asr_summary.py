#!/usr/bin/env python3
"""Экстрактивное саммари поверх карточки доверия (asr.summary/1).

Стадия ничего не придумывает и не использует LLM:

* «Качество записи» целиком рендерится кодом из чисел trust.json — байт-стабильно;
* «О чём встреча», «Решения», «Задачи» — это предложения транскрипта, выбранные
  по ключевым маркерам (решили, договорились, нужно, задача, сделаем, отвечает),
  с таймкодами; если маркеров нет — честно пишется «не найдено»;
* «Спорные места» — окна согласия с dice < 0.35 и uncovered_all из trust.json.

Guard (`guard`) проверяет готовый markdown до записи: каждое число обязано
присутствовать в trust.json (±0.01) либо быть таймкодом; под каждой задачей —
цитата ≤ 120 символов, найденная в транскрипте; запрещены формулировки
«точность N%». Единственное расширение — производные числа, которые сам код
считает из чисел trust.json (`uncovered_s` = сумма uncovered_all); они
перечислены в summary.json → derived_numbers вместе с формулой.

Стадия работает только при verdict ok|partial. failed → summary.md не пишется,
причина уходит в summary.rejected.md, exit 3. blocked → exit 4. Провал guard →
summary.rejected.md, код SUMMARY_GUARD, exit 4.

Exit-коды: 0 ok, 2 partial, 3 failed, 4 blocked/guard, 1 внутренняя, 64 аргументы.

Usage:
  asr_summary.py --trust gate/trust.json --transcript whisper.json --out ./summary
  asr_summary.py --trust trust.json --transcript rec.txt --segments whisper.json --out ./out
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = "asr.summary/1"
TRUST_SCHEMA = "asr.trust/2"

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_PARTIAL = 2
EXIT_FAILED = 3
EXIT_BLOCKED = 4
EXIT_USAGE = 64

SUMMARY_TITLE = "# Саммари встречи"
SECTION_QUALITY = "## Качество записи"
SECTION_OVERVIEW = "## О чём встреча"
SECTION_DECISIONS = "## Решения"
SECTION_TASKS = "## Задачи"
SECTION_DISPUTES = "## Спорные места"
SECTIONS = (SECTION_QUALITY, SECTION_OVERVIEW, SECTION_DECISIONS, SECTION_TASKS, SECTION_DISPUTES)

PARTIAL_PREFIX = "⚠ Качество: partial"
NOT_FOUND = "не найдено"

MAX_QUOTE_CHARS = 120
MAX_OVERVIEW = 5
MAX_ITEMS = 8
OVERVIEW_MIN_CHARS = 25
DICE_LOW = 0.35
NUM_TOL = 0.01

DECISION_MARKERS = ("решил", "решен", "решить", "договор", "согласова", "утверд", "постанов",
                    "приняли решен", "выбрали")
TASK_MARKERS = ("нужно", "надо", "задач", "сдела", "подготов", "отвеча", "ответствен", "провер", "поруч", "дедлайн", "срок")

RU_STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть
надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто
этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее кажется сейчас
были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через
эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше
чуть том нельзя такой им более всегда конечно всю между""".split())

# Таймкод: mm:ss, h:mm:ss, с необязательным «~» (приблизительный, из --segments).
TIMECODE_SPAN = re.compile(r"[~≈]?\b\d{1,3}:\d{2}(?::\d{2})?\b")
NUM_SPAN = re.compile(r"(?<![\w.,])-?\d+(?:[.,]\d+)?(?![\w])")
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
FAKE_PRECISION = re.compile(r"(?i)(точност\w*|accuracy)[^\n]{0,40}?(?:\d|%|процент)")


class SummaryError(Exception):
    """Ошибка стадии с машинным кодом в тексте."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- utils


def is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fmt_num(value) -> str:
    """Число для markdown: без хвостовых нулей, точка как разделитель."""
    if not is_num(value):
        return "н/д"
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        return "н/д"
    s = f"{f:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def timecode(seconds, approx: bool = False) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    core = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
    return ("~" if approx else "") + core


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def norm_for_match(text: str) -> str:
    return norm_ws(text).lower().replace("ё", "е")


def strip_timecodes(text: str) -> str:
    return TIMECODE_SPAN.sub(" ", str(text))


def clip_text(text: str, limit: int) -> str:
    t = norm_ws(text)
    if len(t) <= limit:
        return t
    cut = t[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:—-") + "…"


def quote_for(text: str, limit: int = MAX_QUOTE_CHARS) -> str:
    """Цитата — точный срез текста (без «…», иначе перестанет быть подстрокой)."""
    t = norm_ws(text)
    if len(t) <= limit:
        return t
    cut = t[:limit]
    if " " in cut:
        word_cut = cut[: cut.rfind(" ")]
        if len(word_cut) >= limit // 2:
            return word_cut
    return cut


def span_prefix(item: dict) -> str:
    if not is_num(item.get("start")):
        return ""
    approx = bool(item.get("approx"))
    start = timecode(item["start"], approx)
    if not is_num(item.get("end")) or float(item["end"]) <= float(item["start"]):
        return f"[{start}]"
    return f"[{start}–{timecode(item['end'], approx)}]"


def marker_regex(markers) -> re.Pattern:
    body = "|".join(sorted({re.escape(m) for m in markers}, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{body})", re.IGNORECASE)


DECISION_RE = marker_regex(DECISION_MARKERS)
TASK_RE = marker_regex(TASK_MARKERS)


# ----------------------------------------------------------------------- loading


def read_json(path) -> object:
    """Прочитать JSON любого типа с внятной ошибкой."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SummaryError("FILE_MISSING", f"нет файла: {p}") from exc
    except OSError as exc:
        raise SummaryError("FILE_UNREADABLE", f"не читается {p}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SummaryError("JSON_INVALID", f"{p} не разобран как JSON: {exc}") from exc


def load_json(path: Path) -> dict:
    data = read_json(path)
    if not isinstance(data, dict):
        raise SummaryError("JSON_SHAPE", f"{path}: ожидался объект, получено {type(data).__name__}")
    return data


def load_trust(path) -> dict:
    trust = load_json(Path(path))
    if "verdict" not in trust:
        raise SummaryError("TRUST_NO_VERDICT", "trust.json без поля verdict")
    return trust


def normalize_verdict(trust: dict) -> str:
    verdict = str(trust.get("verdict") or "").strip().lower()
    if verdict in {"ok", "partial", "failed", "blocked"}:
        return verdict
    raise SummaryError("TRUST_VERDICT", f"неизвестный verdict: {trust.get('verdict')!r}")


def reason_codes(trust: dict) -> list[str]:
    codes: list[str] = []
    for reason in trust.get("reasons") or []:
        if isinstance(reason, dict):
            code = str(reason.get("code") or "").strip()
            if code and code not in codes:
                codes.append(code)
    return codes


def _segments_from_payload(payload) -> list[dict]:
    if isinstance(payload, dict):
        rows = payload.get("segments")
        if rows is None:
            rows = payload.get("windows")
    else:
        rows = payload
    out: list[dict] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        text = norm_ws(row.get("text") or "")
        if not text:
            continue
        start, end = row.get("start"), row.get("end")
        bounds = row.get("b") or [None, None]
        if start is None:
            start = bounds[0] if isinstance(bounds, (list, tuple)) and bounds else None
        if end is None:
            end = bounds[1] if isinstance(bounds, (list, tuple)) and len(bounds) > 1 else None
        out.append({
            "start": float(start) if is_num(start) else None,
            "end": float(end) if is_num(end) else None,
            "text": text,
        })
    out.sort(key=lambda r: (r["start"] is None, r["start"] or 0.0))
    return out


def align_to_segments(units: list[dict], segments: list[dict]) -> list[dict]:
    """Разложить плоский текст на таймлайн сегментов пропорционально символам.

    Нужен только когда транскрипт — обычный текст, а таймкоды берутся из
    --segments: это приближение, поэтому такие таймкоды помечаются «~».
    """
    if not segments or not units:
        return units
    total_chars = sum(len(u["text"]) for u in units) or 1
    total_dur = sum(max(0.0, (s["end"] or 0.0) - (s["start"] or 0.0)) for s in segments)
    ends: list[float] = []
    acc = 0.0
    for seg in segments:
        acc += max(0.0, (seg["end"] or 0.0) - (seg["start"] or 0.0))
        ends.append(acc / total_dur if total_dur else 0.0)
    out: list[dict] = []
    pos = 0
    for unit in units:
        mid = (pos + len(unit["text"]) / 2) / total_chars
        index = next((i for i, edge in enumerate(ends) if mid <= edge), len(segments) - 1)
        seg = segments[index]
        out.append({**unit, "start": seg["start"], "end": seg["end"], "approx": True})
        pos += len(unit["text"])
    return out


def split_sentences(text: str) -> list[str]:
    return [part for part in (norm_ws(p) for p in SENT_SPLIT.split(str(text))) if part]


def load_transcript(transcript_path, segments_path=None) -> tuple[list[dict], str]:
    """Вернуть (предложения, полный текст транскрипта).

    Таймкоды берутся из JSON-транскрипта, а если он плоский — из --segments
    (приблизительно, с пометкой «~»). Без таймкодов предложения идут без префикса.
    """
    tpath = Path(transcript_path)
    try:
        raw = tpath.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SummaryError("FILE_MISSING", f"нет файла: {tpath}") from exc
    except OSError as exc:
        raise SummaryError("FILE_UNREADABLE", f"не читается {tpath}: {exc}") from exc

    units: list[dict]
    if tpath.suffix.lower() in {".json", ".jsonl"}:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SummaryError("JSON_INVALID", f"{tpath} не разобран как JSON: {exc}") from exc
        units = _segments_from_payload(payload)
        if not units:
            raise SummaryError("TRANSCRIPT_EMPTY", f"{tpath}: нет сегментов с непустым text")
    else:
        lines = [norm_ws(line) for line in raw.splitlines()]
        units = [{"start": None, "end": None, "text": sentence}
                 for line in lines if line and not line.startswith("#")
                 for sentence in split_sentences(line)]
        if not units:
            raise SummaryError("TRANSCRIPT_EMPTY", f"{tpath}: пустой транскрипт")

    if segments_path is not None:
        extra = _segments_from_payload(read_json(segments_path))
        if extra and all(u.get("start") is None for u in units):
            units = align_to_segments(units, extra)

    sentences: list[dict] = []
    for unit in units:
        for part in split_sentences(unit["text"]):
            sentences.append({
                "start": unit.get("start"),
                "end": unit.get("end"),
                "text": part,
                "approx": bool(unit.get("approx")),
            })
    if not sentences:
        raise SummaryError("TRANSCRIPT_EMPTY", f"{tpath}: не удалось выделить предложения")
    transcript_text = "\n".join(u["text"] for u in units)
    return sentences, norm_ws(transcript_text)


# ---------------------------------------------------------------------- numbers


def _collect_numeric_leaves(node, out: set[float]) -> None:
    if is_num(node):
        out.add(round(float(node), 6))
        return
    if isinstance(node, dict):
        for value in node.values():
            _collect_numeric_leaves(value, out)
    elif isinstance(node, list):
        for value in node:
            _collect_numeric_leaves(value, out)


def trust_numbers(trust: dict) -> list[float]:
    """Все числа trust.json: числовые поля + числа в тексте причин и gate_version."""
    out: set[float] = set()
    _collect_numeric_leaves(trust, out)
    texts = [trust.get("gate_version")]
    texts += [r.get("detail") for r in trust.get("reasons") or [] if isinstance(r, dict)]
    for text in texts:
        if isinstance(text, str):
            for token in NUM_SPAN.findall(text):
                try:
                    out.add(round(float(token.replace(",", ".")), 6))
                except ValueError:
                    continue
    return sorted(out)


def derive_numbers(trust: dict) -> dict[str, dict]:
    """Числа, которые код считает сам из чисел trust.json (с формулой).

    uncovered_s — суммарная длительность uncovered_all, а если этого поля нет —
    сумма speech_gaps по движкам. В asr.trust/2 отдельного uncovered_s нет,
    поэтому сумма фиксируется здесь явно, чтобы guard и verify могли её
    пересчитать; при полном отсутствии данных значение None («н/д»).
    """
    explicit = trust.get("uncovered_s")
    if is_num(explicit):
        return {"uncovered_s": {"value": round(float(explicit), 6),
                                "formula": "trust.uncovered_s", "derived": False}}

    intervals: list[dict] = []
    formula = "sum(uncovered_all[].end - uncovered_all[].start)"
    if "uncovered_all" in trust:
        # поле есть (пусть и пустое) — это и есть канонический источник, как в asr_pipeline.uncovered_s
        intervals = [iv for iv in trust.get("uncovered_all") or [] if isinstance(iv, dict)]
    else:
        coverage = trust.get("coverage") if isinstance(trust.get("coverage"), dict) else {}
        for engine in sorted(coverage):
            payload = coverage[engine]
            if isinstance(payload, dict):
                intervals += [iv for iv in payload.get("speech_gaps") or [] if isinstance(iv, dict)]
        formula = "sum(coverage[*].speech_gaps[].end - start)"

    total = 0.0
    measured = 0
    for interval in intervals:
        if is_num(interval.get("start")) and is_num(interval.get("end")):
            total += max(0.0, float(interval["end"]) - float(interval["start"]))
            measured += 1
    if not measured and "uncovered_all" not in trust:
        return {"uncovered_s": {"value": None,
                                "formula": "в trust.json нет uncovered_all и speech_gaps",
                                "derived": True}}
    return {"uncovered_s": {"value": round(total, 2), "formula": formula, "derived": True}}


def allowed_values(trust: dict, numbers=None, derived=None) -> list[float]:
    nums = trust_numbers(trust) if numbers is None else list(numbers)
    der = derive_numbers(trust) if derived is None else dict(derived)
    values = {round(float(n), 6) for n in nums}
    values |= {round(float(v["value"]), 6) for v in der.values() if is_num(v.get("value"))}
    return sorted(values)


def number_allowed(token: str, allowed: list[float], tol: float = NUM_TOL) -> bool:
    try:
        value = float(str(token).replace(",", "."))
    except ValueError:
        return False
    return any(abs(value - a) <= tol + 1e-9 for a in allowed)


def numbers_in(text: str) -> list[str]:
    return NUM_SPAN.findall(strip_timecodes(text))


def text_is_safe(text: str, allowed: list[float]) -> bool:
    """Предложение можно цитировать, только если все его числа уже есть в trust.json."""
    return all(number_allowed(tok, allowed) for tok in numbers_in(text))


def quality_numbers(trust: dict, derived=None) -> list[float]:
    """Числа, реально опубликованные в блоке «Качество записи» (для numbers_sha256)."""
    der = derive_numbers(trust) if derived is None else derived
    found: set[float] = set()
    for line in quality_lines(trust, der):
        for token in numbers_in(line):
            try:
                found.add(round(float(token.replace(",", ".")), 6))
            except ValueError:
                continue
    return sorted(found)


def numbers_sha256(numbers: list[float]) -> str:
    payload = json.dumps(numbers, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- quality block


def primary_coverage(trust: dict) -> tuple[dict, str]:
    cov = trust.get("coverage") or {}
    if not isinstance(cov, dict) or not cov:
        return {}, str(trust.get("primary") or "")
    primary = trust.get("primary")
    if primary in cov and isinstance(cov[primary], dict):
        return cov[primary], str(primary)
    name = sorted(cov)[0]
    value = cov[name]
    return (value if isinstance(value, dict) else {}), str(name)


def quality_lines(trust: dict, derived: dict | None = None) -> list[str]:
    der = derive_numbers(trust) if derived is None else derived
    verdict = normalize_verdict(trust)
    codes = reason_codes(trust)
    cov, primary = primary_coverage(trust)
    union = trust.get("coverage_union") if isinstance(trust.get("coverage_union"), dict) else {}
    audio = trust.get("audio_metrics") or {}
    activity = trust.get("activity") or {}
    agreement = trust.get("agreement") or {}
    lid = trust.get("language_id") or {}

    lines = [f"- вердикт: `{verdict}`"]
    lines.append("- коды причин: " + (", ".join(f"`{c}`" for c in codes) if codes else "нет"))

    if is_num(audio.get("duration_s")):
        lines.append(f"- длительность: {fmt_num(audio['duration_s'])} с")

    audio_parts = []
    for key, label in (("rms_dbfs", "RMS"), ("peak_dbfs", "пик")):
        if is_num(audio.get(key)):
            audio_parts.append(f"{label} {fmt_num(audio[key])} dBFS")
    if is_num(audio.get("clip_fraction")):
        audio_parts.append(f"клиппинг {fmt_num(audio['clip_fraction'])}")
    if is_num(audio.get("band_share_below_1khz")):
        audio_parts.append(f"доля энергии в НЧ-полосе {fmt_num(audio['band_share_below_1khz'])}")
    if audio_parts:
        lines.append("- аудио: " + ", ".join(audio_parts))

    activity_parts = []
    if is_num(activity.get("energy_active_s")):
        activity_parts.append(f"энергия {fmt_num(activity['energy_active_s'])} с")
    if is_num(activity.get("vad_active_s")):
        activity_parts.append(f"VAD {fmt_num(activity['vad_active_s'])} с")
    if is_num(activity.get("noise_floor_dbfs")):
        activity_parts.append(f"шумовой пол {fmt_num(activity['noise_floor_dbfs'])} dBFS")
    if activity_parts:
        lines.append("- активность: " + ", ".join(activity_parts))

    coverage_parts = []
    union_value = union.get("coverage_union") if is_num(union.get("coverage_union")) else None
    if is_num(union_value):
        coverage_parts.append(f"union {fmt_num(union_value)}")
    elif is_num(cov.get("coverage_union")):
        coverage_parts.append(f"union {fmt_num(cov['coverage_union'])}")
    elif is_num(cov.get("coverage")):
        coverage_parts.append(fmt_num(cov["coverage"]))
    covered = union.get("covered_s") if is_num(union.get("covered_s")) else cov.get("covered_s")
    active = union.get("active_s") if is_num(union.get("active_s")) else cov.get("union_active_s")
    if is_num(covered) and is_num(active):
        coverage_parts.append(f"covered {fmt_num(covered)} с из {fmt_num(active)} с")
    elif is_num(covered):
        coverage_parts.append(f"covered {fmt_num(covered)} с")
    if is_num(cov.get("tail_uncovered_s")):
        coverage_parts.append(f"хвост без текста {fmt_num(cov['tail_uncovered_s'])} с")
    if coverage_parts:
        lines.append("- покрытие" + (f" «{primary}»" if primary else "") + ": " + ", ".join(coverage_parts))

    uncovered = der["uncovered_s"]["value"]
    if is_num(uncovered):
        lines.append(f"- uncovered_s: {fmt_num(uncovered)} (сумма интервалов без текста)")
    else:
        lines.append("- uncovered_s: н/д (в trust.json нет uncovered_all)")

    if is_num(agreement.get("mean_dice")):
        pair = agreement.get("pair") or []
        pair_text = f" («{pair[0]}» / «{pair[1]}»)" if isinstance(pair, list) and len(pair) >= 2 else ""
        lines.append(f"- согласие: среднее dice {fmt_num(agreement['mean_dice'])}{pair_text}")
    else:
        lines.append("- согласие: нет данных (второй движок не сравнивался)")

    if lid.get("language"):
        lid_parts = [str(lid["language"])]
        if is_num(lid.get("probability")):
            lid_parts.append(f"вероятность {fmt_num(lid['probability'])}")
        if lid.get("method"):
            lid_parts.append(f"метод {lid['method']}")
        lines.append("- язык: " + ", ".join(lid_parts))
    else:
        lines.append("- язык: нет данных LID")

    lines.append("- саммари: экстрактивное, LLM не использовался")
    return lines


# ------------------------------------------------------------------- extraction


def content_tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[а-яёa-z]{4,}", text.lower()) if w not in RU_STOP]


def pick_overview(sentences: list[dict]) -> list[dict]:
    if not sentences:
        return []
    candidates = [s for s in sentences if len(s["text"]) >= OVERVIEW_MIN_CHARS] or list(sentences)
    scored = sorted(
        enumerate(candidates),
        key=lambda pair: (-len(set(content_tokens(pair[1]["text"]))), -len(pair[1]["text"]), pair[0]),
    )
    return [candidates[i] for i, _ in sorted(scored[:MAX_OVERVIEW], key=lambda pair: pair[0])]


def pick_marked(sentences: list[dict], regex: re.Pattern, limit: int = MAX_ITEMS) -> list[dict]:
    picked: list[dict] = []
    seen: set[str] = set()
    for sentence in sentences:
        if not regex.search(sentence["text"]):
            continue
        key = norm_for_match(sentence["text"])
        if key in seen:
            continue
        seen.add(key)
        picked.append(sentence)
        if len(picked) >= limit:
            break
    return picked


def dispute_lines(trust: dict) -> list[str]:
    lines: list[str] = []
    agreement = trust.get("agreement") or {}
    pair = agreement.get("pair") or []
    pair_text = f"; {' / '.join(str(p) for p in pair)}" if isinstance(pair, list) and pair else ""
    for window in agreement.get("windows") or []:
        if not isinstance(window, dict):
            continue
        dice = window.get("dice")
        if not is_num(dice) or float(dice) >= DICE_LOW:
            continue
        if not (is_num(window.get("start")) and is_num(window.get("end"))):
            continue
        lines.append(f"- [{timecode(window['start'])}–{timecode(window['end'])}] "
                     f"расхождение движков, dice {fmt_num(dice)}{pair_text}")
    for interval in trust.get("uncovered_all") or []:
        if not isinstance(interval, dict):
            continue
        if not (is_num(interval.get("start")) and is_num(interval.get("end"))):
            continue
        lines.append(f"- [{timecode(interval['start'])}–{timecode(interval['end'])}] "
                     f"активность без текста (uncovered_all)")
    return lines[:MAX_ITEMS]


def render_summary(trust: dict, sentences: list[dict], transcript_text: str) -> tuple[str, dict]:
    verdict = normalize_verdict(trust)
    codes = reason_codes(trust)
    der = derive_numbers(trust)
    allowed = allowed_values(trust, derived=der)

    safe = [s for s in sentences if text_is_safe(s["text"], allowed)]
    dropped = len(sentences) - len(safe)

    lines: list[str] = [SUMMARY_TITLE, ""]
    if verdict == "partial":
        lines += [f"{PARTIAL_PREFIX} — коды: " + (", ".join(codes) if codes else "не указаны"), ""]

    lines += [SECTION_QUALITY, ""] + quality_lines(trust, der) + [""]

    decisions = pick_marked(safe, DECISION_RE)
    tasks = pick_marked(safe, TASK_RE)
    decisions_all = pick_marked(sentences, DECISION_RE)
    tasks_all = pick_marked(sentences, TASK_RE)
    marked_keys = {norm_for_match(s["text"]) for s in decisions + tasks}

    overview_pool = [s for s in safe if norm_for_match(s["text"]) not in marked_keys]
    if len(overview_pool) < 3:
        overview_pool = safe
    overview = pick_overview(overview_pool)
    lines += [SECTION_OVERVIEW, ""]
    if overview:
        lines += [f"- {span_prefix(s)} {clip_text(s['text'], 240)}".strip() for s in overview]
    elif dropped and sentences:
        lines.append(f"- {NOT_FOUND} — все предложения содержат числа вне trust.json")
    else:
        lines.append(f"- {NOT_FOUND} — в транскрипте нет пригодных предложений")
    lines.append("")

    lines += [SECTION_DECISIONS, ""]
    if decisions:
        lines += [f"- {span_prefix(s)} {clip_text(s['text'], 240)}".strip() for s in decisions]
    elif decisions_all:
        lines.append(f"- {NOT_FOUND} — подходящие предложения содержат числа вне trust.json")
    else:
        lines.append(f"- {NOT_FOUND} — нет предложений с маркерами решений (решили, договорились, согласовали)")
    lines.append("")

    lines += [SECTION_TASKS, ""]
    if tasks:
        for task in tasks:
            lines.append(f"- {span_prefix(task)} {clip_text(task['text'], 240)}".strip())
            lines.append(f"  > «{quote_for(task['text'])}»")
    elif tasks_all:
        lines.append(f"- {NOT_FOUND} — подходящие предложения содержат числа вне trust.json")
    else:
        lines.append(f"- {NOT_FOUND} — нет предложений с маркерами задач (нужно, задача, сделаем, отвечает)")
    lines.append("")

    disputes = dispute_lines(trust)
    lines += [SECTION_DISPUTES, ""]
    lines += disputes or [f"- {NOT_FOUND} — расхождений движков и непокрытых участков нет"]
    lines.append("")

    meta = {
        "sections": {
            "quality": True,
            "overview": bool(overview),
            "decisions": bool(decisions),
            "tasks": bool(tasks),
            "disputes": bool(disputes),
        },
        "sentences_total": len(sentences),
        "sentences_safe": len(safe),
        "skipped": dropped,
        "numbers": quality_numbers(trust, der),
        "derived": der,
    }
    return "\n".join(lines).rstrip("\n") + "\n", meta


# ------------------------------------------------------------------------ guard


def section_slice(md: str, heading: str) -> list[str] | None:
    lines = md.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return None
    out: list[str] = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        out.append(line)
    return out


def body_numbers(md: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for lineno, line in enumerate(md.splitlines(), start=1):
        for token in NUM_SPAN.findall(strip_timecodes(line)):
            found.append((lineno, token))
    return found


def check_caveat(md: str, verdict: str, codes: list[str]) -> tuple[bool, list[str]]:
    nonempty = [line for line in md.splitlines() if line.strip()]
    if not nonempty or nonempty[0].strip() != SUMMARY_TITLE:
        return False, [f"первая строка должна быть «{SUMMARY_TITLE}»"]
    if verdict != "partial":
        return True, []
    if len(nonempty) < 2 or not nonempty[1].startswith(PARTIAL_PREFIX):
        return False, [f"первая строка после заголовка должна начинаться с «{PARTIAL_PREFIX}»"]
    missing = [code for code in codes if code not in nonempty[1]]
    if missing:
        return False, ["в предупреждении нет кодов причин: " + ", ".join(missing)]
    return True, []


def check_tasks(md: str, transcript_norm: str) -> tuple[bool, list[str]]:
    section = section_slice(md, SECTION_TASKS)
    if section is None:
        return False, [f"нет раздела «{SECTION_TASKS}»"]
    problems: list[str] = []
    entries = [line.strip() for line in section]
    index = 0
    while index < len(entries):
        line = entries[index]
        if line.startswith("- ") and not line.startswith(f"- {NOT_FOUND}"):
            quote_line = ""
            probe = index + 1
            while probe < len(entries) and not entries[probe]:
                probe += 1
            if probe < len(entries):
                quote_line = entries[probe]
            match = re.match(r"^>\s*«(.+?)»\s*$", quote_line)
            if not match:
                problems.append(f"задача без цитаты: {clip_text(line[2:], 60)}")
            else:
                quote = match.group(1)
                if len(quote) > MAX_QUOTE_CHARS:
                    problems.append(f"цитата длиннее {MAX_QUOTE_CHARS} символов: {clip_text(quote, 40)}")
                if norm_for_match(quote) not in transcript_norm:
                    problems.append(f"цитата не найдена в транскрипте: {clip_text(quote, 60)}")
            index = probe
        index += 1
    return (not problems), problems


def guard(md: str, trust: dict, transcript_text: str, numbers=None, derived=None) -> dict:
    """Проверить markdown до записи. Пустой violations == можно публиковать."""
    allowed = allowed_values(trust, numbers=numbers, derived=derived)
    verdict = normalize_verdict(trust)
    codes = reason_codes(trust)
    violations: list[str] = []

    numbers_ok = True
    checked = 0
    for lineno, token in body_numbers(md):
        checked += 1
        if not number_allowed(token, allowed):
            numbers_ok = False
            violations.append(f"строка {lineno}: число {token} отсутствует в trust.json")

    phrases_ok = True
    for lineno, line in enumerate(md.splitlines(), start=1):
        hit = FAKE_PRECISION.search(line)
        if hit:
            phrases_ok = False
            violations.append(f"строка {lineno}: запрещённая формулировка точности «{clip_text(hit.group(0), 60)}»")

    sections_ok = True
    for heading in SECTIONS:
        if section_slice(md, heading) is None:
            sections_ok = False
            violations.append(f"нет раздела «{heading}»")

    caveat_present, caveat_problems = check_caveat(md, verdict, codes)
    violations += caveat_problems

    tasks_quoted, task_problems = check_tasks(md, norm_for_match(transcript_text))
    violations += task_problems

    return {
        "ok": bool(numbers_ok and phrases_ok and sections_ok and caveat_present and tasks_quoted),
        "numbers_ok": numbers_ok,
        "tasks_quoted": tasks_quoted,
        "caveat_present": caveat_present,
        "phrases_ok": phrases_ok,
        "sections_ok": sections_ok,
        "numbers_checked": checked,
        "violations": violations,
    }


# --------------------------------------------------------------------- stage IO


def refusal_markdown(title: str, reasons: list[str]) -> str:
    lines = [f"# {title}", ""] + [f"- {reason}" for reason in reasons] + [""]
    return "\n".join(lines)


def summary_card(status: str, verdict: str, codes: list[str], guard_result: dict,
                 meta: dict | None, numbers: list[float], derived: dict, extra: list[str] | None = None) -> dict:
    guard_result = guard_result or {}
    card = {
        "schema": SCHEMA,
        "status": status,
        "trusted": bool(status == "ok" and verdict == "ok" and guard_result.get("ok")),
        "verdict": verdict,
        "codes": codes,
        "guard": {
            "numbers_ok": bool(guard_result.get("numbers_ok")),
            "tasks_quoted": bool(guard_result.get("tasks_quoted")),
            "caveat_present": bool(guard_result.get("caveat_present")),
            "phrases_ok": bool(guard_result.get("phrases_ok")),
            "sections_ok": bool(guard_result.get("sections_ok")),
            "numbers_checked": int(guard_result.get("numbers_checked") or 0),
            "violations": list(guard_result.get("violations") or []),
        },
        "numbers": numbers,
        "numbers_sha256": numbers_sha256(numbers),
        "derived_numbers": derived,
        "sections": (meta or {}).get("sections", {}),
        "files": [],
    }
    if extra:
        card["warnings"] = list(extra)
    return card


def summarize(trust_path, transcript_path, segments_path, out_dir) -> dict:
    """Выполнить стадию. Возвращает карточку summary.json (с полем exit для CLI)."""
    trust = load_trust(trust_path)
    verdict = normalize_verdict(trust)
    codes = reason_codes(trust)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = trust_numbers(trust)
    derived = derive_numbers(trust)
    warnings: list[str] = []
    if trust.get("schema") and trust.get("schema") != TRUST_SCHEMA:
        warnings.append(f"trust schema {trust.get('schema')} вместо {TRUST_SCHEMA}")

    if verdict in {"failed", "blocked"}:
        exit_code = EXIT_FAILED if verdict == "failed" else EXIT_BLOCKED
        stage_code = "SUMMARY_REFUSED_FAILED" if verdict == "failed" else "SUMMARY_REFUSED_BLOCKED"
        reason = ("стадия саммари работает только при verdict ok|partial: "
                  f"trust.verdict={verdict}")
        reasons = [f"код: {stage_code}", f"причина: {reason}"]
        if codes:
            reasons.append("коды trust: " + ", ".join(codes))
        reasons.append("summary.md не создан намеренно: качество записи не позволяет пересказывать текст")
        (out / "summary.rejected.md").write_text(
            refusal_markdown("Саммари не создано", reasons), encoding="utf-8")
        card = summary_card("refused", verdict, codes + [stage_code], {}, None,
                            quality_numbers(trust, derived), derived, warnings)
        card["reason"] = reason
        card["files"] = ["summary.rejected.md"]
        card["exit"] = exit_code
        (out / "summary.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        return card

    sentences, transcript_text = load_transcript(transcript_path, segments_path)
    md, meta = render_summary(trust, sentences, transcript_text)
    result = guard(md, trust, transcript_text, numbers, derived)

    if not result["ok"]:
        rejected = md.rstrip("\n") + "\n\n## Причины отказа\n\n" + \
            "\n".join(f"- {violation}" for violation in result["violations"]) + "\n"
        (out / "summary.rejected.md").write_text(rejected, encoding="utf-8")
        card = summary_card("refused", verdict, codes + ["SUMMARY_GUARD"], result, meta,
                            meta.get("numbers", numbers), derived, warnings)
        card["reason"] = "guard отклонил саммари: SUMMARY_GUARD"
        card["files"] = ["summary.rejected.md"]
        card["exit"] = EXIT_BLOCKED
        (out / "summary.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        return card

    (out / "summary.md").write_text(md, encoding="utf-8")
    exit_code = EXIT_OK if verdict == "ok" else EXIT_PARTIAL
    card = summary_card("ok", verdict, codes, result, meta, meta.get("numbers", numbers), derived, warnings)
    card["summary_sha256"] = hashlib.sha256(md.encode("utf-8")).hexdigest()
    card["files"] = ["summary.md"]
    card["exit"] = exit_code
    (out / "summary.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    return card


# ------------------------------------------------------------------------- CLI


class SummaryArgumentParser(argparse.ArgumentParser):
    def error(self, message):  # argparse по умолчанию выходит с 2, контракт требует 64
        print(json.dumps({"schema": SCHEMA, "error": message, "code": "USAGE"}, ensure_ascii=False))
        raise SystemExit(EXIT_USAGE)


def parse_args(argv=None) -> argparse.Namespace:
    parser = SummaryArgumentParser(description="Экстрактивное саммари ASR с guard по trust.json")
    parser.add_argument("--trust", required=True, metavar="PATH")
    parser.add_argument("--transcript", required=True, metavar="PATH")
    parser.add_argument("--segments", default=None, metavar="PATH")
    parser.add_argument("--out", required=True, metavar="DIR")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        card = summarize(args.trust, args.transcript, args.segments, args.out)
    except SummaryError as exc:
        print(json.dumps({"schema": SCHEMA, "status": "error", "code": exc.code,
                          "error": exc.message, "exit": EXIT_INTERNAL}, ensure_ascii=False))
        return EXIT_INTERNAL
    except OSError as exc:
        print(json.dumps({"schema": SCHEMA, "status": "error", "code": "IO_ERROR",
                          "error": str(exc), "exit": EXIT_INTERNAL}, ensure_ascii=False))
        return EXIT_INTERNAL
    print(json.dumps({key: card[key] for key in
                      ("schema", "status", "verdict", "exit", "codes", "guard", "numbers_sha256")
                      if key in card}, ensure_ascii=False))
    return int(card["exit"])


if __name__ == "__main__":
    sys.exit(main())
