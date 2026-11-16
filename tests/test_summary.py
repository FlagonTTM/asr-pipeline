#!/usr/bin/env python3
"""Тесты стадии саммари: контракт, guard, экстрактивность, детерминизм.

Сеть и ML не используются: trust.json и транскрипт синтетические.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asr_summary as S  # noqa: E402

SCRIPT = ROOT / "asr_summary.py"
PYTHON = sys.executable
TIMECODE_LINE = re.compile(r"^- \[(?:~)?\d{2}:\d{2}–(?:~)?\d{2}:\d{2}\] ")


# --------------------------------------------------------------------- helpers


def trust_card(verdict="ok", reasons=None, dice=0.74, uncovered=None, **override):
    """Синтетическая карточка asr.trust/2 с самосогласованными числами."""
    reasons = list(reasons or [])
    uncovered = list(uncovered if uncovered is not None else [])
    card = {
        "schema": "asr.trust/2",
        "verdict": verdict,
        "primary": "whisper",
        "gate_version": "2.0",
        "reasons": reasons,
        "audio_metrics": {
            "duration_s": 178.25,
            "rms_dbfs": -41.58,
            "peak_dbfs": -13.09,
            "clip_fraction": 0.0,
            "band_share_below_1khz": 0.751,
            "rms_per_10s": [-40.2, -42.6],
        },
        "activity": {"energy_active_s": 157.6, "vad_active_s": 31.2, "noise_floor_dbfs": -83.8},
        "language_id": {"language": "ru", "probability": 0.971, "method": "whisper"},
        "coverage": {
            "whisper": {
                "coverage": 0.87,
                "coverage_union": 0.87,
                "covered_s": 150.0,
                "union_active_s": 157.6,
                "tail_uncovered_s": 7.1,
                "speech_gaps": [{"start": 116.0, "end": 148.0}],
            }
        },
        "coverage_union": {"covered_s": 150.0, "active_s": 157.6, "coverage_union": 0.87},
        "uncovered_all": uncovered,
        "agreement": {
            "mean_dice": dice,
            "windows": [{"start": 0, "end": 20, "dice": dice}, {"start": 20, "end": 40, "dice": 0.81}],
            "pair": ["whisper", "kts"],
        },
        "config": {"vad": "energy", "lid": "whisper", "thresholds": {}},
        "verdict_legacy": verdict,
    }
    card.update(override)
    return card


TRANSCRIPT = {
    "schema": "asr.segments/1",
    "audio": {"path": "/abs/audio.wav", "sha256": "0" * 64, "duration_s": 178.25, "sr": 16000, "channels": 1},
    "segments": [
        {"start": 0.0, "end": 7.46, "text": "Потом была гидроэнергия."},
        {"start": 7.46, "end": 22.26, "text": "Они не берут разнообразия."},
        {"start": 22.26, "end": 28.47,
         "text": "То есть мы говорим о том, что мы высаживаем деревья и получаем за это токены."},
        {"start": 28.47, "end": 33.13, "text": "Решили, что пилот запускаем в мае."},
        {"start": 33.13, "end": 48.18, "text": "Нужно подготовить смету до пятницы."},
        {"start": 48.18, "end": 74.02, "text": "Задачу по мониторингу отвечает Мария."},
        {"start": 74.02, "end": 81.81, "text": "Договорились, что отчёт будет еженедельным."},
        {"start": 81.81, "end": 92.26, "text": "Сделаем выгрузку по токенам."},
    ],
}

PLAIN_TRANSCRIPT = "\n".join(
    [
        "Потом была гидроэнергия.",
        "Они не берут разнообразия.",
        "То есть мы говорим о том, что мы высаживаем деревья и получаем за это токены.",
        "Решили, что пилот запускаем в мае.",
        "Нужно подготовить смету до пятницы.",
    ]
)


def write_inputs(tmp_path, trust=None, transcript=None, segments=None, name="in"):
    base = Path(tmp_path) / name
    base.mkdir(parents=True, exist_ok=True)
    trust_path = base / "trust.json"
    trust_path.write_text(json.dumps(trust or trust_card(), ensure_ascii=False), encoding="utf-8")
    if isinstance(transcript, str):
        transcript_path = base / "transcript.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
    else:
        transcript_path = base / "segments.json"
        transcript_path.write_text(json.dumps(transcript or TRANSCRIPT, ensure_ascii=False), encoding="utf-8")
    segments_path = None
    if segments is not None:
        segments_path = base / "timing.json"
        segments_path.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    return trust_path, transcript_path, segments_path


def run_stage(tmp_path, trust=None, transcript=None, segments=None, name="in", out="out"):
    trust_path, transcript_path, segments_path = write_inputs(tmp_path, trust, transcript, segments, name)
    out_dir = Path(tmp_path) / name / out
    argv = ["--trust", str(trust_path), "--transcript", str(transcript_path), "--out", str(out_dir)]
    if segments_path is not None:
        argv += ["--segments", str(segments_path)]
    code = S.main(argv)
    return code, out_dir


def read_md(out_dir):
    return (Path(out_dir) / "summary.md").read_text(encoding="utf-8")


def read_card(out_dir):
    return json.loads((Path(out_dir) / "summary.json").read_text(encoding="utf-8"))


def section(md, heading):
    lines = md.splitlines()
    start = lines.index(heading)
    body = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return [line for line in body if line.strip()]


# ----------------------------------------------------------------------- tests


def test_ok_run_has_all_sections_and_exit_zero(tmp_path):
    code, out_dir = run_stage(tmp_path, trust_card("ok", dice=0.74), name="ok")
    assert code == 0
    md = read_md(out_dir)
    assert md.startswith(S.SUMMARY_TITLE)
    for heading in S.SECTIONS:
        assert heading in md
    assert "⚠" not in md
    card = read_card(out_dir)
    assert card["schema"] == "asr.summary/1"
    assert card["status"] == "ok"
    assert card["trusted"] is True
    assert card["verdict"] == "ok"
    assert card["guard"]["numbers_ok"] is True
    assert card["guard"]["tasks_quoted"] is True
    assert card["guard"]["caveat_present"] is True
    assert card["numbers_sha256"] == S.numbers_sha256(card["numbers"])
    assert card["files"] == ["summary.md"]


def test_partial_puts_warning_right_after_title(tmp_path):
    reasons = [
        {"code": "SPEECH_GAP", "severity": "warn", "detail": "2 участка без текста",
         "intervals": [{"start": 116.0, "end": 148.0}]},
        {"code": "LOW_AGREEMENT", "severity": "warn", "detail": "согласие 0.28", "intervals": []},
    ]
    code, out_dir = run_stage(
        tmp_path, trust_card("partial", reasons=reasons, dice=0.28,
                             uncovered=[{"start": 116.0, "end": 148.0}]),
        name="partial",
    )
    assert code == 2
    lines = [line for line in read_md(out_dir).splitlines() if line.strip()]
    assert lines[0] == S.SUMMARY_TITLE
    assert lines[1].startswith("⚠ Качество: partial")
    assert "SPEECH_GAP" in lines[1] and "LOW_AGREEMENT" in lines[1]


def test_failed_does_not_write_summary_and_exits_three(tmp_path):
    reasons = [{"code": "LOW_COVERAGE", "severity": "block", "detail": "покрытие 0.42", "intervals": []}]
    code, out_dir = run_stage(tmp_path, trust_card("failed", reasons=reasons), name="failed")
    assert code == 3
    assert not (out_dir / "summary.md").exists()
    rejected = (out_dir / "summary.rejected.md").read_text(encoding="utf-8")
    assert "LOW_COVERAGE" in rejected
    assert "summary.md не создан" in rejected
    card = read_card(out_dir)
    assert card["status"] == "refused"
    assert card["trusted"] is False
    assert "SUMMARY_REFUSED_FAILED" in card["codes"]
    assert card["exit"] == 3


def test_blocked_exits_four_without_summary(tmp_path):
    code, out_dir = run_stage(tmp_path, trust_card("blocked", reasons=[
        {"code": "LANGUAGE_MISMATCH", "severity": "block", "detail": "LID=en p=0.97", "intervals": []}
    ]), name="blocked")
    assert code == 4
    assert not (out_dir / "summary.md").exists()
    card = read_card(out_dir)
    assert card["status"] == "refused"
    assert "SUMMARY_REFUSED_BLOCKED" in card["codes"]


def test_quality_block_renders_only_trust_numbers(tmp_path):
    reasons = [{"code": "SPEECH_GAP", "severity": "warn", "detail": "участок без текста",
                "intervals": [{"start": 116.0, "end": 148.0}]}]
    trust = trust_card("partial", reasons=reasons, dice=0.28, uncovered=[{"start": 116.0, "end": 148.0}])
    code, out_dir = run_stage(tmp_path, trust, name="quality")
    assert code == 2
    quality = "\n".join(section(read_md(out_dir), S.SECTION_QUALITY))
    allowed = S.allowed_values(trust)
    tokens = S.numbers_in(quality)
    assert tokens, "в блоке качества должны быть числа"
    for token in tokens:
        assert S.number_allowed(token, allowed), f"число {token} отсутствует в trust.json"
    for expected in ("178.25", "-41.58", "157.6", "0.87", "7.1", "0.28", "0.971", "32"):
        assert expected in quality
    assert "uncovered_s" in quality


def test_quality_block_is_byte_stable_across_runs(tmp_path):
    trust = trust_card("partial", reasons=[{"code": "SPEECH_GAP", "severity": "warn",
                                            "detail": "участок без текста", "intervals": []}],
                       dice=0.28, uncovered=[{"start": 116.0, "end": 148.0}])
    _, first = run_stage(tmp_path, trust, name="stable_a")
    _, second = run_stage(tmp_path, trust, name="stable_b")
    block_a = section(read_md(first), S.SECTION_QUALITY)
    block_b = section(read_md(second), S.SECTION_QUALITY)
    assert block_a == block_b
    assert hashlib.sha256("\n".join(block_a).encode()).hexdigest() == \
        hashlib.sha256("\n".join(block_b).encode()).hexdigest()


def test_extractive_sections_use_transcript_sentences_with_timecodes(tmp_path):
    code, out_dir = run_stage(tmp_path, trust_card("ok", dice=0.74), name="extractive")
    assert code == 0
    md = read_md(out_dir)
    decisions = section(md, S.SECTION_DECISIONS)
    tasks = section(md, S.SECTION_TASKS)
    assert any("Решили, что пилот запускаем в мае" in line for line in decisions)
    assert any("Договорились, что отчёт будет еженедельным" in line for line in decisions)
    assert any("Нужно подготовить смету до пятницы" in line for line in tasks)
    for line in decisions:
        assert TIMECODE_LINE.match(line), line
    task_bullets = [line for line in tasks if line.startswith("- ")]
    assert task_bullets
    for line in task_bullets:
        assert TIMECODE_LINE.match(line), line


def test_every_task_has_quote_from_transcript(tmp_path):
    code, out_dir = run_stage(tmp_path, trust_card("ok", dice=0.74), name="quotes")
    assert code == 0
    md = read_md(out_dir)
    tasks = section(md, S.SECTION_TASKS)
    transcript = " ".join(seg["text"] for seg in TRANSCRIPT["segments"])
    quotes = []
    for index, line in enumerate(tasks):
        if not line.startswith("- "):
            continue
        quote_line = tasks[index + 1]
        match = re.match(r"^>\s*«(.+?)»$", quote_line.strip())
        assert match, f"под задачей нет цитаты: {line}"
        quote = match.group(1)
        assert len(quote) <= S.MAX_QUOTE_CHARS
        assert quote in transcript
        quotes.append(quote)
    assert quotes


def test_no_markers_reports_not_found_honestly(tmp_path):
    transcript = {
        "schema": "asr.segments/1",
        "segments": [
            {"start": 0.0, "end": 7.0, "text": "Потом была гидроэнергия."},
            {"start": 7.0, "end": 15.0, "text": "Они не берут разнообразия."},
            {"start": 15.0, "end": 24.0, "text": "То есть мы высаживаем деревья и получаем токены."},
        ],
    }
    code, out_dir = run_stage(tmp_path, trust_card("ok", dice=0.74), transcript, name="nomarkers")
    assert code == 0
    md = read_md(out_dir)
    for heading in (S.SECTION_DECISIONS, S.SECTION_TASKS):
        body = " ".join(section(md, heading))
        assert "не найдено" in body
        assert not re.search(r"\d", body.replace(heading, "")), body


def test_disputes_come_from_agreement_windows_and_uncovered(tmp_path):
    trust = trust_card("partial", reasons=[{"code": "LOW_AGREEMENT", "severity": "warn",
                                            "detail": "согласие 0.28", "intervals": []}],
                       dice=0.28, uncovered=[{"start": 116.0, "end": 148.0}])
    code, out_dir = run_stage(tmp_path, trust, name="disputes")
    assert code == 2
    body = section(read_md(out_dir), S.SECTION_DISPUTES)
    text = "\n".join(body)
    assert "[00:00–00:20]" in text
    assert "dice 0.28" in text
    assert "[01:56–02:28]" in text
    assert "uncovered_all" in text


def test_guard_catches_invented_number_182(tmp_path):
    trust = trust_card("ok", dice=0.74)
    sentences, transcript_text = S.load_transcript(*write_inputs(tmp_path, trust)[1:2])
    md, _ = S.render_summary(trust, sentences, transcript_text)
    tampered = md.replace(S.SECTION_DECISIONS, "- Бюджет проекта 182 тыс. рублей\n\n" + S.SECTION_DECISIONS)
    result = S.guard(tampered, trust, transcript_text)
    assert result["numbers_ok"] is False
    assert result["ok"] is False
    assert any("182" in violation for violation in result["violations"])


def test_guard_catches_fake_precision_phrase(tmp_path):
    trust = trust_card("ok", dice=0.74)
    sentences, transcript_text = S.load_transcript(*write_inputs(tmp_path, trust)[1:2])
    md, _ = S.render_summary(trust, sentences, transcript_text)
    tampered = md.replace(S.SECTION_DECISIONS, "- Точность 95% по бенчмарку\n\n" + S.SECTION_DECISIONS)
    result = S.guard(tampered, trust, transcript_text)
    assert result["phrases_ok"] is False
    assert result["ok"] is False
    assert any("точност" in violation.lower() for violation in result["violations"])


def test_guard_rejects_quote_not_in_transcript(tmp_path):
    trust = trust_card("ok", dice=0.74)
    sentences, transcript_text = S.load_transcript(*write_inputs(tmp_path, trust)[1:2])
    md, _ = S.render_summary(trust, sentences, transcript_text)
    lines = md.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("  > «"):
            lines[index] = "  > «Такого предложения в транскрипте нет»"
            break
    result = S.guard("\n".join(lines), trust, transcript_text)
    assert result["tasks_quoted"] is False
    assert result["ok"] is False
    assert any("цитата не найдена" in violation for violation in result["violations"])


def test_cli_guard_failure_writes_rejected_and_exits_four(tmp_path, monkeypatch):
    trust = trust_card("ok", dice=0.74)
    real_render = S.render_summary

    def tampered_render(trust_card_, sentences, transcript_text):
        md, meta = real_render(trust_card_, sentences, transcript_text)
        return md.replace(S.SECTION_TASKS, "## Задачи\n\n- выдуманная задача 182 раза\n"), meta

    monkeypatch.setattr(S, "render_summary", tampered_render)
    trust_path, transcript_path, _ = write_inputs(tmp_path, trust, name="guardfail")
    out_dir = Path(tmp_path) / "guardfail" / "out"
    code = S.main(["--trust", str(trust_path), "--transcript", str(transcript_path), "--out", str(out_dir)])
    assert code == 4
    assert not (out_dir / "summary.md").exists()
    rejected = (out_dir / "summary.rejected.md").read_text(encoding="utf-8")
    assert "## Причины отказа" in rejected
    card = read_card(out_dir)
    assert card["status"] == "refused"
    assert "SUMMARY_GUARD" in card["codes"]
    assert card["guard"]["numbers_ok"] is False


def test_two_cli_runs_produce_identical_sha256(tmp_path):
    trust = trust_card("partial", reasons=[{"code": "SPEECH_GAP", "severity": "warn",
                                            "detail": "участок без текста", "intervals": []}],
                       dice=0.28, uncovered=[{"start": 116.0, "end": 148.0}])
    hashes = []
    cards = []
    for name in ("det_a", "det_b"):
        trust_path, transcript_path, _ = write_inputs(tmp_path, trust, name=name)
        out_dir = Path(tmp_path) / name / "out"
        proc = subprocess.run(
            [PYTHON, str(SCRIPT), "--trust", str(trust_path), "--transcript", str(transcript_path),
             "--out", str(out_dir)],
            capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 2, proc.stdout + proc.stderr
        payload = (out_dir / "summary.md").read_bytes()
        hashes.append(hashlib.sha256(payload).hexdigest())
        card = read_card(out_dir)
        assert card["summary_sha256"] == hashes[-1]
        cards.append(json.dumps(card, ensure_ascii=False, sort_keys=True))
    assert hashes[0] == hashes[1]
    assert cards[0] == cards[1]


def test_usage_error_exits_64():
    proc = subprocess.run([PYTHON, str(SCRIPT), "--transcript", "x.json", "--out", "y"],
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 64
    assert json.loads(proc.stdout)["code"] == "USAGE"


def test_plain_text_transcript_with_segments_gets_approx_timecodes(tmp_path):
    segments = [{"start": 0.0, "end": 30.0, "text": "a"}, {"start": 30.0, "end": 60.0, "text": "b"}]
    code, out_dir = run_stage(tmp_path, trust_card("ok", dice=0.74), PLAIN_TRANSCRIPT,
                              segments=segments, name="plain")
    assert code == 0
    md = read_md(out_dir)
    decisions = section(md, S.SECTION_DECISIONS)
    assert decisions and all(line.startswith("- [~") for line in decisions)
    assert "Решили, что пилот запускаем в мае" in decisions[0]


def test_missing_trust_file_exits_internal(tmp_path):
    out_dir = Path(tmp_path) / "missing" / "out"
    code = S.main(["--trust", str(tmp_path / "nope.json"), "--transcript", str(tmp_path / "x.json"),
                   "--out", str(out_dir)])
    assert code == 1


def test_gate_shaped_trust_without_union_fields_still_passes_guard(tmp_path):
    """asr_gate.py не пишет coverage_union/uncovered_all: стадия обязана не врать и не падать."""
    gate_like = {
        "verdict": "partial",
        "reasons": [{"code": "LOW_LEVEL", "detail": "RMS -41.58 dBFS"}],
        "audio_metrics": {"duration_s": 178.25, "rms_dbfs": -41.58, "peak_dbfs": -13.09,
                          "clip_fraction": 0.0, "band_share_below_1khz": 0.751, "rms_per_10s": [-40.2]},
        "activity": {"energy_active_s": 157.6, "vad_active_s": 31.2, "noise_floor_dbfs": -83.8},
        "language_id": {"language": "ru", "probability": 0.971, "method": "whisper"},
        "coverage": {"whisper": {"covered_s": 150.0, "coverage": 0.87, "last_segment_end": 171.1,
                                 "tail_uncovered_s": 7.1,
                                 "speech_gaps": [{"start": 116.0, "end": 148.0}]}},
        "agreement": {"mean_dice": 0.28, "windows": [{"start": 0, "end": 20, "dice": 0.2}],
                      "pair": ["whisper", "kts"]},
    }
    code, out_dir = run_stage(tmp_path, gate_like, name="gateshape")
    assert code == 2
    md = read_md(out_dir)
    quality = "\n".join(section(md, S.SECTION_QUALITY))
    assert "uncovered_s: 32" in quality  # сумма speech_gaps
    assert "union" not in quality  # coverage_union не выдумываем
    assert S.guard(md, gate_like, " ".join(seg["text"] for seg in TRANSCRIPT["segments"]))["ok"] is True


def test_missing_uncovered_data_reports_not_available(tmp_path):
    trust = trust_card("ok", dice=0.74)
    trust["coverage"]["whisper"].pop("speech_gaps")
    del trust["uncovered_all"]
    code, out_dir = run_stage(tmp_path, trust, name="nodata")
    assert code == 0
    quality = "\n".join(section(read_md(out_dir), S.SECTION_QUALITY))
    assert "uncovered_s: н/д" in quality
    assert read_card(out_dir)["derived_numbers"]["uncovered_s"]["value"] is None


def test_empty_uncovered_all_is_zero_like_pipeline_metric(tmp_path):
    trust = trust_card("ok", dice=0.74)
    trust["uncovered_all"] = []
    code, out_dir = run_stage(tmp_path, trust, name="zerogap")
    assert code == 0
    quality = "\n".join(section(read_md(out_dir), S.SECTION_QUALITY))
    assert "uncovered_s: 0" in quality
    assert read_card(out_dir)["derived_numbers"]["uncovered_s"]["value"] == 0.0


def test_sentences_with_unknown_numbers_are_not_quoted(tmp_path):
    transcript = {
        "schema": "asr.segments/1",
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "Нужно подготовить 12 отчётов до пятницы."},
            {"start": 10.0, "end": 20.0, "text": "Решили сократить бюджет."},
        ],
    }
    code, out_dir = run_stage(tmp_path, trust_card("ok", dice=0.74), transcript, name="unsafe")
    assert code == 0
    md = read_md(out_dir)
    tasks = " ".join(section(md, S.SECTION_TASKS))
    assert "12" not in tasks
    assert "не найдено" in tasks
    decisions = section(md, S.SECTION_DECISIONS)
    assert any("Решили сократить бюджет" in line for line in decisions)


def test_no_transcript_numbers_leak_into_body(tmp_path):
    """Сквозная проверка: guard не находит в готовом summary.md ни одного лишнего числа."""
    trust = trust_card("partial", reasons=[{"code": "SPEECH_GAP", "severity": "warn",
                                            "detail": "участок без текста", "intervals": []}],
                       dice=0.28, uncovered=[{"start": 116.0, "end": 148.0}])
    code, out_dir = run_stage(tmp_path, trust, name="leak")
    assert code == 2
    md = read_md(out_dir)
    result = S.guard(md, trust, " ".join(seg["text"] for seg in TRANSCRIPT["segments"]))
    assert result["ok"] is True, result["violations"]
    assert result["numbers_checked"] > 0
