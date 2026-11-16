"""Тесты asr_gate v2: без сети и без ML.

Покрывают ровно те места, где 1.x врал на реальных данных:
  * покрытие = union(сегменты) ∩ активность, а не сумма длительностей (сумма давала 1.0 при дырах);
  * дыры = непокрытая активность >= 1.0 с (пересечение), а не «вложенность сегмента в интервал»;
  * --primary фиксирует вердикт: порядок --transcript его не меняет;
  * --vad off даёт vad_active_s=None и не выдумывает VAD_UNRELIABLE;
  * .jsonl читается построчно;
  * неверные аргументы -> exit 64;
  * severity (warn|block) и intervals в reasons, block -> failed, схема asr.trust/2.

LID/VAD-тесты (faster-whisper) помечены skip: ML в тестах не гоняем.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asr_gate as g  # noqa: E402

SR = 16000
FIX = ROOT / "fixtures"
REASON_CODES = {"LANGUAGE_MISMATCH", "LOW_LEVEL", "CLIPPING", "VAD_UNRELIABLE",
                "TAIL_UNCOVERED", "LOW_COVERAGE", "SPEECH_GAP", "LOW_AGREEMENT", "LOW_INFORMATION"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tone(n: int, freq: float = 440.0, amp: float = 0.25) -> np.ndarray:
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_wav(path, spans, duration, freq=440.0, amp=0.25) -> Path:
    """Синтетика: синус в spans, цифровая тишина в остальном."""
    x = np.zeros(int(round(duration * SR)), dtype=np.float32)
    for lo, hi in spans:
        a, b = int(round(lo * SR)), min(int(round(hi * SR)), len(x))
        x[a:b] = tone(b - a, freq, amp)
    sf.write(str(path), x, SR)
    return Path(path)


def write_json(path, segments) -> Path:
    Path(path).write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")
    return Path(path)


def run_gate_cli(tmp_path, wav, transcripts, *extra, out_name="out"):
    """Полный прогон CLI; возвращает (exit_code, trust.json|None)."""
    out = tmp_path / out_name
    args = ["--audio", str(wav), "--out", str(out)]
    for name, segs in transcripts.items():
        args += ["--transcript", f"{name}={write_json(tmp_path / f'{out_name}-{name}.json', segs)}"]
    args += [str(e) for e in extra]
    code = g.main(args)
    trust = out / "trust.json"
    return code, (json.loads(trust.read_text(encoding="utf-8")) if trust.exists() else None)


# ---------------------------------------------------------------------------
# аудио-метрики и активность
# ---------------------------------------------------------------------------
def test_audio_metrics_sine_and_silence():
    x = np.concatenate([tone(4 * SR), np.zeros(2 * SR, np.float32),
                        tone(3 * SR), np.zeros(SR, np.float32)])
    m = g.audio_metrics(x, SR)

    assert m["duration_s"] == 10.0
    assert m["rms_dbfs"] == pytest.approx(-16.6, abs=0.4)
    assert m["peak_dbfs"] == pytest.approx(-12.04, abs=0.3)
    assert m["clip_fraction"] == 0.0
    assert 0.9 < m["band_share_below_1khz"] <= 1.0     # 440 Гц
    assert len(m["rms_per_10s"]) == 1                   # окна по 10 с, а не по 1 с
    assert m["rms_per_10s"][0] == pytest.approx(-16.6, abs=0.4)


def test_audio_metrics_band_share_tracks_frequency():
    low = g.audio_metrics(np.concatenate([tone(3 * SR, 440.0), np.zeros(SR, np.float32)]), SR)
    high = g.audio_metrics(np.concatenate([tone(3 * SR, 4000.0), np.zeros(SR, np.float32)]), SR)
    assert low["band_share_below_1khz"] > 0.9
    assert high["band_share_below_1khz"] < 0.1


def test_energy_activity_two_stretches():
    x = np.concatenate([tone(4 * SR), np.zeros(2 * SR, np.float32),
                        tone(3 * SR), np.zeros(SR, np.float32)])
    e = g.energy_activity(x, SR)

    assert e["active_s"] == pytest.approx(7.0, abs=0.2)
    assert len(e["intervals"]) == 2
    assert e["intervals"][0]["start"] == pytest.approx(0.0, abs=0.05)
    assert e["intervals"][0]["end"] == pytest.approx(4.0, abs=0.1)
    assert e["intervals"][1]["start"] == pytest.approx(6.0, abs=0.1)
    assert e["intervals"][1]["end"] == pytest.approx(9.0, abs=0.1)
    assert e["noise_floor_dbfs"] < -100                 # цифровая тишина


def test_energy_activity_silence_only():
    e = g.energy_activity(np.zeros(2 * SR, np.float32), SR)
    assert e["active_s"] == 0.0
    assert e["intervals"] == []


def test_energy_activity_without_pauses_uses_absolute_floor():
    """Без пауз перцентильный пол сидит в речи: абсолютный порог спасает знаменатель."""
    e = g.energy_activity(tone(8 * SR), SR)
    assert e["active_s"] >= 7.5
    assert len(e["intervals"]) == 1


# ---------------------------------------------------------------------------
# union vs sum и геометрия дыр
# ---------------------------------------------------------------------------
def test_interval_algebra():
    assert g.merge_intervals([(0, 5), (3, 8), (10, 12)]) == [(0, 8), (10, 12)]
    assert g.union_length([(0, 5), (1, 3)]) == 5.0
    assert g.intersect([(0, 5), (8, 10)], [(4, 9)]) == [(4, 5), (8, 9)]
    assert g.subtract([(0, 10)], [(2, 4), (8, 20)]) == [(0, 2), (4, 8)]


def test_coverage_is_union_not_sum():
    """Три сегмента с перекрытием: сумма 15 с, union 10 с."""
    segs = [{"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 10.0, "text": "b"}]
    c = g.coverage(segs, 10.0, [{"start": 0.0, "end": 10.0}])
    assert c["covered_s"] == 10.0
    assert c["coverage_union"] == 1.0
    assert sum(s["end"] - s["start"] for s in segs) == 15.0   # сумма врёт, union — нет


def test_empty_segments_are_counted_without_changing_interval_coverage():
    segs = [{"start": 0.0, "end": 5.0, "text": "полезная реплика"},
            {"start": 5.0, "end": 7.0, "text": ""},
            {"start": 7.0, "end": 7.1, "text": "   "},
            {"start": 7.1, "end": 8.0, "text": None},
            {"start": 8.0, "end": 10.0, "text": "Да."}]
    filled = [dict(s, text="полезная реплика") for s in segs]
    active = [{"start": 0.0, "end": 10.0}]
    actual = g.coverage(segs, 10.0, active)
    reference = g.coverage(filled, 10.0, active)
    assert actual["empty_segments"] == 3
    assert reference["empty_segments"] == 0
    assert actual["coverage_union"] == 1.0
    assert actual["covered_s"] == 10.0
    assert actual["speech_gaps"] == []
    assert dict(actual, empty_segments=0) == reference


def test_gap_is_intersection_not_containment():
    """Старый тест «сегмент содержит интервал активности» давал ложные дыры."""
    segs = [{"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 10.0, "end": 20.0, "text": "b"}]
    c = g.coverage(segs, 20.0, [{"start": 0.0, "end": 20.0}])
    assert c["speech_gaps"] == []           # ни один сегмент не содержит 0–20, но union покрывает
    assert c["coverage_union"] == 1.0

    holes = g.coverage([{"start": 0.0, "end": 10.0, "text": "a"},
                        {"start": 12.0, "end": 20.0, "text": "b"}],
                       20.0, [{"start": 0.0, "end": 20.0}])
    assert holes["speech_gaps"] == [{"start": 10.0, "end": 12.0}]
    assert holes["coverage_union"] == pytest.approx(0.9, abs=0.001)


def test_short_holes_ignored():
    segs = [{"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 10.5, "end": 20.0, "text": "b"}]   # дыра 0.5 с < 1.0
    c = g.coverage(segs, 20.0, [{"start": 0.0, "end": 20.0}])
    assert c["speech_gaps"] == []


def test_coverage_none_when_no_activity():
    c = g.coverage([{"start": 0.0, "end": 5.0, "text": "a"}], 10.0, [])
    assert c["coverage_union"] is None
    assert c["covered_s"] == 0.0
    assert c["speech_gaps"] == []


def test_empty_transcript_blocks():
    """Пустой транскрипт — это не «нет данных», а нулевое покрытие."""
    c = g.coverage([], 10.0, [{"start": 0.0, "end": 10.0}])
    assert c["coverage_union"] == 0.0
    assert c["tail_uncovered_s"] == 10.0
    assert c["speech_gaps"] == [{"start": 0.0, "end": 10.0}]
    assert g.verdict_of(g.build_reasons(**_facts(coverage=c))) == "failed"


def test_coverage_clips_to_duration():
    c = g.coverage([{"start": 0.0, "end": 30.0, "text": "a"}], 10.0, [{"start": 0.0, "end": 10.0}])
    assert c["covered_s"] == 10.0
    assert c["coverage_union"] == 1.0
    assert c["last_segment_end"] == 10.0


def test_fixture_whisper_union_vs_sum():
    """Сумма двух проходов вдвое больше union: старая формула дала бы 1.0, union — 0.90."""
    wh = g.load_segments(FIX / "whisper.json")
    assert len(wh) == 16
    seg_sum = sum(s["end"] - s["start"] for s in wh)
    activity = [{"start": 0.0, "end": 178.25}]

    one = g.coverage(wh, 178.25, activity)
    twice = g.coverage(wh + wh, 178.25, activity)

    assert seg_sum == pytest.approx(160.89, abs=0.05)
    assert one["covered_s"] == pytest.approx(160.9, abs=0.1)
    assert twice["covered_s"] == one["covered_s"]          # union не растёт от дублей
    assert twice["coverage_union"] == one["coverage_union"] == pytest.approx(0.903, abs=0.002)

    naive_sum = min(2 * seg_sum / 178.25, 1.0)             # формула 1.x
    assert naive_sum == 1.0 and twice["coverage_union"] < 0.95

    # реальные непокрытые участки: пауза 165.05–168.19 и хвост после 170.09
    assert {"start": 165.05, "end": 168.19} in one["speech_gaps"]
    assert one["tail_uncovered_s"] == pytest.approx(8.2, abs=0.05)


def test_fixture_kts_gap_by_intersection():
    """Стенд КТС: реальная дыра 129–148 не должна дробиться на ложные куски."""
    kt = g.load_segments(FIX / "kts.json")
    assert len(kt) == 10
    c = g.coverage(kt, 178.25, [{"start": 0.0, "end": 178.25}])

    assert {"start": 129.0, "end": 148.0} in c["speech_gaps"]
    # стык 16/17: вложенность дала бы 15.55–17.75 (2.2 с), пересечение — ровно 1.0 с
    assert {"start": 16.0, "end": 17.0} in c["speech_gaps"]
    assert all(not (a["start"] < 150 and a["end"] > 150) for a in c["speech_gaps"])
    assert c["coverage_union"] == pytest.approx(149.0 / 178.25, abs=0.002)


# ---------------------------------------------------------------------------
# причины: severity и вердикт
# ---------------------------------------------------------------------------
def _facts(**over):
    base = {
        "metrics": {"rms_dbfs": -30.0, "clip_fraction": 0.0},
        "activity": {"energy_active_s": 100.0, "vad_active_s": 100.0},
        "coverage": {"coverage_union": 1.0, "tail_uncovered_s": 0.0, "speech_gaps": [],
                     "last_segment_end": 100.0},
        "lid": {"language": "ru", "probability": 1.0, "method": "declared"},
        "language": "ru",
        "agreement": None,
        "energy_intervals": [{"start": 0.0, "end": 100.0}],
    }
    base.update(over)
    return base


def test_clean_facts_give_ok():
    assert g.build_reasons(**_facts()) == []
    assert g.verdict_of([]) == "ok"


def test_vad_underreport_is_warn():
    reasons = g.build_reasons(**_facts(activity={"energy_active_s": 100.0, "vad_active_s": 10.0}))
    assert [(r["code"], r["severity"]) for r in reasons] == [("VAD_UNRELIABLE", "warn")]
    assert g.verdict_of(reasons) == "partial"


def test_vad_off_none_does_not_raise_unreliable():
    reasons = g.build_reasons(**_facts(activity={"energy_active_s": 100.0, "vad_active_s": None}))
    assert reasons == []


@pytest.mark.parametrize("cov,severity,expected", [(0.95, None, "ok"), (0.80, "warn", "partial"),
                                                   (0.40, "block", "failed")])
def test_low_coverage_severity_and_verdict(cov, severity, expected):
    facts = _facts()
    facts["coverage"] = dict(facts["coverage"], coverage_union=cov)
    reasons = g.build_reasons(**facts)
    by_code = {r["code"]: r for r in reasons}
    assert by_code.get("LOW_COVERAGE", {}).get("severity") == severity
    assert g.verdict_of(reasons) == expected


def test_low_coverage_block_makes_verdict_failed():
    facts = _facts()
    facts["coverage"] = dict(facts["coverage"], coverage_union=0.5)
    report_verdict = g.verdict_of(g.build_reasons(**facts))
    assert report_verdict == "failed"
    reasons = g.build_reasons(**facts)
    assert g.verdict_legacy_of(reasons) == "failed"


def test_tail_and_gap_reasons_carry_intervals():
    facts = _facts()
    facts["coverage"] = dict(facts["coverage"], coverage_union=0.95, tail_uncovered_s=12.0,
                             last_segment_end=88.0,
                             speech_gaps=[{"start": 40.0, "end": 42.0},
                                          {"start": 88.0, "end": 100.0}])
    reasons = g.build_reasons(**facts)
    codes = {r["code"]: r for r in reasons}
    assert codes["TAIL_UNCOVERED"]["severity"] == "warn"
    assert codes["TAIL_UNCOVERED"]["intervals"] == [{"start": 88.0, "end": 100.0}]
    assert codes["SPEECH_GAP"]["intervals"] == facts["coverage"]["speech_gaps"]
    assert codes["SPEECH_GAP"]["severity"] == "warn"


def test_language_mismatch_blocks():
    facts = _facts(lid={"language": "en", "probability": 0.97, "method": "whisper"})
    reasons = g.build_reasons(**facts)
    assert [(r["code"], r["severity"]) for r in reasons] == [("LANGUAGE_MISMATCH", "block")]
    assert g.verdict_of(reasons) == "failed"


def test_language_off_never_mismatches():
    facts = _facts(lid={"language": "ru", "probability": None, "method": "off"})
    assert g.build_reasons(**facts) == []


def test_low_agreement_intervals():
    facts = _facts(agreement={"mean_dice": 0.2, "pair": ["a", "b"],
                              "windows": [{"start": 0, "end": 20, "dice": 0.1},
                                          {"start": 20, "end": 40, "dice": 0.9}]})
    reasons = g.build_reasons(**facts)
    agr = [r for r in reasons if r["code"] == "LOW_AGREEMENT"][0]
    assert agr["intervals"] == [{"start": 0.0, "end": 20.0}]
    assert agr["severity"] == "warn"


@pytest.mark.parametrize("worst,severity", [(0.34, "warn"), (0.35, None), (0.36, None)])
def test_low_agreement_threshold_applies_to_worst_peer_not_global_mean(worst, severity):
    agr = {"mean_dice": 0.8, "pair": ["a", "b"], "per_peer": {"b": 1.0, "c": worst},
           "windows": [{"start": 0, "end": 20, "dice": 1.0}],
           "windows_per_peer": {"b": [{"start": 0, "end": 20, "dice": 1.0}],
                                "c": [{"start": 0, "end": 20, "dice": worst}]}}
    reasons = g.build_reasons(**_facts(agreement=agr))
    assert [(r["code"], r["severity"]) for r in reasons] == (
        [] if severity is None else [("LOW_AGREEMENT", severity)])


# ---------------------------------------------------------------------------
# загрузка транскриптов: json / jsonl
# ---------------------------------------------------------------------------
def test_load_fixture_json():
    assert len(g.load_segments(FIX / "whisper.json")) == 16
    assert len(g.load_segments(FIX / "kts.json")) == 10


def test_load_jsonl_line_by_line(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join([
        json.dumps({"start": 0, "end": 1, "text": "раз"}),
        "",
        json.dumps({"start": 1.5, "end": 2.5, "text": "два"}),
    ]), encoding="utf-8")
    segs = g.load_segments(p)
    assert [s["start"] for s in segs] == [0.0, 1.5]
    assert [s["end"] for s in segs] == [1.0, 2.5]


def test_load_jsonl_with_envelope_and_pretty_json(tmp_path):
    env = {"segments": [{"start": 0, "end": 1, "text": "a"},
                        {"start": 1, "end": 2, "text": "b"}]}
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    assert len(g.load_segments(p)) == 2


def test_load_json_that_is_jsonl(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"start": 0, "end": 1, "text": "a"}) + "\n" +
                 json.dumps({"start": 1, "end": 2, "text": "b"}), encoding="utf-8")
    assert [s["start"] for s in g.load_segments(p)] == [0.0, 1.0]


def test_load_skips_broken_rows(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"segments": [
        {"start": 0, "end": 1, "text": "ok"},
        {"start": 2, "end": 2, "text": "zero"},
        {"start": "x", "end": 3, "text": "bad"},
        {"b": [4, 5], "text": "bounds"},
    ]}), encoding="utf-8")
    assert [s["start"] for s in g.load_segments(p)] == [0.0, 4.0]


def test_load_unsupported_suffix(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="LOAD_FAILED"):
        g.load_segments(p)


def test_load_broken_jsonl_raises(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"start":0,"end":1,"text":"a"}\n{broken\n', encoding="utf-8")
    with pytest.raises(ValueError, match="LOAD_FAILED"):
        g.load_segments(p)


# ---------------------------------------------------------------------------
# CLI: вердикт, --primary, --vad off, схема
# ---------------------------------------------------------------------------
WHISPER_LIKE = [{"start": 0.0, "end": 8.8, "text": "потом была гидроэнергия"}]
KTS_LIKE = [{"start": 0.0, "end": 2.0, "text": "потом была гидроэнергия"}]


@pytest.fixture(scope="module")
def speech_wav(tmp_path_factory):
    """12 с: тон 0–9 с, 3 с тишины — шумовой пол оценивается по тишине."""
    d = tmp_path_factory.mktemp("audio")
    return make_wav(d / "tone.wav", [(0, 9)], 12.0)


def test_cli_ok_report_and_schema(tmp_path, speech_wav):
    code, r = run_gate_cli(tmp_path, speech_wav, {"whisper": WHISPER_LIKE},
                           "--vad", "energy", "--lid", "declared")

    assert code == 0
    assert r["schema"] == "asr.trust/2"
    assert r["gate_version"] == "2.0"
    assert r["verdict"] == "ok"
    assert r["primary"] == "whisper"
    for key in ("coverage", "coverage_union", "uncovered_all", "config", "reasons",
                "audio_metrics", "activity", "language_id", "agreement", "verdict_legacy"):
        assert key in r
    assert r["empty_segments_total"] == 0
    assert r["coverage"]["whisper"]["empty_segments"] == 0
    assert r["config"]["vad"] == "energy"
    assert r["config"]["lid"] == "declared"
    assert r["config"]["thresholds"]["min_gap_s"] == 1.0
    assert r["activity"]["vad_active_s"] == r["activity"]["energy_active_s"]
    assert r["coverage_union"]["active_s"] == r["activity"]["energy_active_s"]
    assert r["uncovered_all"] == r["coverage"]["whisper"]["speech_gaps"]
    assert r["language_id"] == {"language": "ru", "probability": 1.0, "method": "declared"}
    assert (tmp_path / "out" / "trust.md").exists()
    assert (tmp_path / "out" / "vocab.md").exists()


def test_report_counts_empty_segments_for_all_transcripts_without_changing_coverage(tmp_path, speech_wav):
    transcripts = {
        "whisper": [{"start": 0.0, "end": 3.0, "text": "полезная реплика"},
                    {"start": 3.0, "end": 6.0, "text": ""},
                    {"start": 6.0, "end": 8.8, "text": "   "}],
        "peer": [{"start": 0.0, "end": 4.0, "text": "другая реплика"},
                 {"start": 4.0, "end": 8.8, "text": None}],
    }
    filled = {name: [dict(s, text="полезная реплика") for s in segs]
              for name, segs in transcripts.items()}
    _, actual = run_gate_cli(tmp_path, speech_wav, transcripts,
                             "--vad", "energy", "--lid", "declared", out_name="empty")
    _, reference = run_gate_cli(tmp_path, speech_wav, filled,
                                "--vad", "energy", "--lid", "declared", out_name="filled")
    assert actual["coverage"]["whisper"]["empty_segments"] == 2
    assert actual["coverage"]["peer"]["empty_segments"] == 1
    assert actual["empty_segments_total"] == 3
    assert reference["empty_segments_total"] == 0
    assert actual["coverage_union"] == reference["coverage_union"]
    for name in transcripts:
        assert dict(actual["coverage"][name], empty_segments=0) == reference["coverage"][name]


def test_transcript_order_does_not_change_verdict(tmp_path, speech_wav):
    a, ra = run_gate_cli(tmp_path, speech_wav, {"whisper": WHISPER_LIKE, "kts": KTS_LIKE},
                         "--primary", "whisper", "--vad", "energy", "--lid", "declared",
                         out_name="a")
    b, rb = run_gate_cli(tmp_path, speech_wav, {"kts": KTS_LIKE, "whisper": WHISPER_LIKE},
                         "--primary", "whisper", "--vad", "energy", "--lid", "declared",
                         out_name="b")

    assert a == b == 0
    assert ra["verdict"] == rb["verdict"] == "ok"
    assert ra["primary"] == rb["primary"] == "whisper"
    assert ra["coverage_union"] == rb["coverage_union"]
    assert ra["reasons"] == rb["reasons"] == []
    assert ra["agreement"]["pair"] == rb["agreement"]["pair"] == ["whisper", "kts"]


@pytest.mark.parametrize("reverse_order", [False, True])
def test_multi_peer_agreement_uses_worst_peer_and_unions_all_low_windows(tmp_path, reverse_order):
    wav = make_wav(tmp_path / "agreement.wav", [(0, 80)], 80.0)
    primary = [{"start": i * 20.0, "end": (i + 1) * 20.0,
                "text": "альфа бета гамма дельта"} for i in range(4)]
    partial = [dict(s, text="другие слова без совпадений" if i == 0 else s["text"])
               for i, s in enumerate(primary)]
    bad = [dict(s, text=s["text"] if i == 2 else "иная фраза полностью отличается")
           for i, s in enumerate(primary)]
    transcripts = {"whisper": primary, "a_good": primary, "b_partial": partial, "c_bad": bad}
    if reverse_order:
        transcripts = dict(reversed(list(transcripts.items())))
    code, report = run_gate_cli(tmp_path, wav, transcripts,
                                "--primary", "whisper", "--vad", "energy", "--lid", "declared")
    agr = report["agreement"]
    assert agr["per_peer"] == {"a_good": 1.0, "b_partial": 0.75, "c_bad": 0.25}
    assert agr["mean_dice"] == 0.67
    assert agr["pair"] == ["whisper", "a_good"]
    assert agr["windows"] == [{"start": i * 20, "end": (i + 1) * 20, "dice": 1.0, "dice_raw": 1.0}
                              for i in range(4)]
    assert code == 2
    assert report["verdict"] == "partial"
    assert [(r["code"], r["severity"]) for r in report["reasons"]] == [("LOW_AGREEMENT", "warn")]
    reason = report["reasons"][0]
    assert "c_bad" in reason["detail"]
    assert reason["intervals"] == [{"start": 0.0, "end": 40.0}, {"start": 60.0, "end": 80.0}]
    # Публичного agreement из trust.json достаточно для повторного построения причины.
    assert g.build_reasons(**_facts(agreement=agr)) == report["reasons"]


@pytest.mark.parametrize("with_good_peer", [False, True])
def test_agreement_threshold_uses_unrounded_dice(tmp_path, speech_wav, with_good_peer):
    primary = [{"start": 0.0, "end": 8.8, "text": " ".join(map(str, range(23)))}]
    peer = [dict(primary[0], text=" ".join(map(str, [*range(8), *range(23, 38)])))]
    transcripts = {"whisper": primary, "z_bad": peer}
    if with_good_peer:
        transcripts["a_good"] = primary
    code, report = run_gate_cli(tmp_path, speech_wav, transcripts,
                                "--vad", "energy", "--lid", "declared")
    agr = report["agreement"]
    assert agr["per_peer"]["z_bad"] == 0.35  # формат старого поля сохранён
    assert code == 2
    assert report["verdict"] == "partial"
    reasons = [r for r in report["reasons"] if r["code"] == "LOW_AGREEMENT"]
    assert len(reasons) == 1
    assert reasons[0]["intervals"] == [{"start": 0.0, "end": 20.0}]
    assert agr["per_peer_raw"]["z_bad"] == pytest.approx(16 / 46)
    assert g.build_reasons(**_facts(agreement=agr)) == reasons


@pytest.mark.parametrize("low_count,severity", [(2, None), (3, "warn"), (5, "warn")])
def test_low_agreement_window_share_threshold_is_inclusive(low_count, severity):
    primary = [{"start": i, "end": i + 1, "text": " ".join(map(str, range(23)))}
               for i in range(10)]
    peer = [dict(s, text=" ".join(map(str, [*range(8), *range(23, 38)])))
            if i < low_count else dict(s) for i, s in enumerate(primary)]
    agr = g.agreement(primary, peer, 1.0, 10.0)
    assert agr["mean_dice"] > 0.35
    reasons = g.build_reasons(**_facts(agreement=agr))
    assert [(r["code"], r["severity"]) for r in reasons] == (
        [] if severity is None else [("LOW_AGREEMENT", severity)])
    assert agr["low_window_share"] == low_count / 10
    if severity:
        assert reasons[0]["intervals"] == [{"start": 0.0, "end": float(low_count)}]


def test_local_low_agreement_is_not_hidden_by_mean_or_good_peer(tmp_path):
    counts = [(14, 4), (10, 0), (10, 2), (10, 5), (10, 8), (10, 8)]
    primary = [{"start": i * 20, "end": (i + 1) * 20,
                "text": " ".join(map(str, range(total)))}
               for i, (total, shared) in enumerate(counts)]
    peer = [dict(s, text=" ".join(map(str, [*range(shared), *range(total, 2 * total - shared)])))
            for s, (total, shared) in zip(primary, counts)]
    wav = make_wav(tmp_path / "local-agreement.wav", [(0, 120)], 120.0)
    code, report = run_gate_cli(tmp_path, wav, {"whisper": primary, "a_good": primary, "b_local": peer},
                                "--vad", "energy", "--lid", "declared")
    agr = report["agreement"]
    assert agr["per_peer"]["b_local"] == 0.43
    assert [w["dice"] for w in agr["windows_per_peer"]["b_local"][:3]] == [0.29, 0.0, 0.2]
    assert code == 2
    assert report["verdict"] == "partial"
    reasons = [r for r in report["reasons"] if r["code"] == "LOW_AGREEMENT"]
    assert len(reasons) == 1
    assert reasons[0]["intervals"] == [{"start": 0.0, "end": 60.0}]
    assert agr["low_window_share"] == 0.5  # худшая доля по peer, не разбавляется хорошим
    assert agr["low_window_share_per_peer"] == {"a_good": 0.0, "b_local": 0.5}
    assert report["config"]["thresholds"]["low_agreement_window_share"] == 0.30
    assert g.build_reasons(**_facts(agreement=agr)) == reasons


def test_multi_peer_markdown_labels_overall_and_individual_scores(tmp_path, speech_wav):
    transcripts = {"whisper": WHISPER_LIKE, "a_good": WHISPER_LIKE,
                   "b_bad": [{"start": 0.0, "end": 8.8, "text": "другие слова"}]}
    _, report = run_gate_cli(tmp_path, speech_wav, transcripts,
                             "--vad", "energy", "--lid", "declared")
    markdown = (tmp_path / "out" / "trust.md").read_text(encoding="utf-8")
    assert "Среднее согласие" in markdown
    for name, dice in report["agreement"]["per_peer"].items():
        assert f"«{name}»: {dice}" in markdown


def test_multi_peer_mean_weights_pairs_equally_not_window_counts(tmp_path):
    wav = make_wav(tmp_path / "pairs.wav", [(0, 40)], 40.0)
    primary = [{"start": 0.0, "end": 20.0, "text": "общая полезная реплика"}]
    peer = primary + [{"start": 20.0, "end": 40.0, "text": "другая реплика"}]
    _, report = run_gate_cli(tmp_path, wav, {"whisper": primary, "a": primary, "b": peer},
                             "--vad", "energy", "--lid", "declared")
    assert report["agreement"]["per_peer"] == {"a": 1.0, "b": 0.5}
    assert report["agreement"]["mean_dice"] == 0.75
    assert "LOW_AGREEMENT" in {r["code"] for r in report["reasons"]}
    assert report["agreement"]["low_window_share"] == 0.5


@pytest.mark.parametrize("other_text,expected_mean", [("", None), ("полезная реплика", 0.0)])
def test_multi_peer_agreement_handles_pairs_without_tokens(tmp_path, speech_wav, other_text, expected_mean):
    blank = [{"start": 0.0, "end": 8.8, "text": ""}]
    peer = [dict(blank[0], text=other_text)]
    _, report = run_gate_cli(tmp_path, speech_wav, {"whisper": blank, "a": blank, "b": peer},
                             "--vad", "energy", "--lid", "declared")
    assert report["agreement"]["per_peer"] == {"a": None, "b": expected_mean}
    assert report["agreement"]["mean_dice"] == expected_mean
    assert report["agreement"]["windows"] == []
    assert ("LOW_AGREEMENT" in {r["code"] for r in report["reasons"]}) == (expected_mean is not None)


def test_default_primary_is_whisper_and_deterministic(tmp_path, speech_wav):
    _, ra = run_gate_cli(tmp_path, speech_wav, {"kts": KTS_LIKE, "whisper": WHISPER_LIKE},
                         "--vad", "energy", "--lid", "declared", out_name="a")
    _, rb = run_gate_cli(tmp_path, speech_wav, {"whisper": WHISPER_LIKE, "kts": KTS_LIKE},
                         "--vad", "energy", "--lid", "declared", out_name="b")
    assert ra["primary"] == rb["primary"] == "whisper"
    assert ra["verdict"] == rb["verdict"]


def test_primary_selects_verdict_failed_on_block(tmp_path, speech_wav):
    code, r = run_gate_cli(tmp_path, speech_wav, {"whisper": WHISPER_LIKE, "kts": KTS_LIKE},
                           "--primary", "kts", "--vad", "energy", "--lid", "declared")
    assert r["primary"] == "kts"
    assert r["verdict"] == "failed"
    assert code == 2                                   # гейт отдаёт только 0/2
    assert {"code": "LOW_COVERAGE", "severity": "block"} in [
        {"code": x["code"], "severity": x["severity"]} for x in r["reasons"]]
    assert r["verdict_legacy"] == "failed"
    assert r["coverage"]["kts"]["coverage_union"] < g.LOW_COVERAGE_BLOCK


def test_vad_off_has_null_vad_active(tmp_path, speech_wav):
    code, r = run_gate_cli(tmp_path, speech_wav, {"whisper": WHISPER_LIKE},
                           "--vad", "off", "--lid", "declared")
    assert r["activity"]["vad_active_s"] is None
    assert r["activity"]["vad"] == "off"
    assert "VAD_UNRELIABLE" not in {x["code"] for x in r["reasons"]}
    assert code in (0, 2)


def test_reasons_contract(tmp_path, speech_wav):
    _, r = run_gate_cli(tmp_path, speech_wav, {"whisper": [{"start": 0.0, "end": 6.5,
                                                            "text": "текст"}]},
                        "--vad", "energy", "--lid", "declared")
    assert r["reasons"], "ожидались причины: покрытие 0.72, дыра 6.5–9 с"
    for reason in r["reasons"]:
        assert reason["code"] in REASON_CODES
        assert reason["severity"] in {"warn", "block"}
        assert isinstance(reason["intervals"], list)
        assert isinstance(reason["detail"], str) and reason["detail"]
    gap = [x for x in r["reasons"] if x["code"] == "SPEECH_GAP"][0]
    assert gap["intervals"] == r["uncovered_all"]
    assert len(r["uncovered_all"]) == 1
    assert r["uncovered_all"][0]["start"] == pytest.approx(6.5, abs=0.01)
    assert r["uncovered_all"][0]["end"] == pytest.approx(9.0, abs=0.1)
    assert r["verdict"] == "partial"
    assert r["coverage_union"]["coverage_union"] == pytest.approx(6.5 / 9.0, abs=0.01)


# ---------------------------------------------------------------------------
# CLI: валидация аргументов -> 64
# ---------------------------------------------------------------------------
def test_empty_invocation_exits_64(capsys):
    assert g.main([]) == 64
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["exit"] == 64 and "BAD_ARGS" in payload["error"]


def test_missing_transcript_exits_64(tmp_path, speech_wav):
    assert g.main(["--audio", str(speech_wav), "--out", str(tmp_path / "o")]) == 64


def test_missing_audio_file_exits_64(tmp_path):
    assert g.main(["--audio", str(tmp_path / "nope.wav"),
                   "--transcript", f"whisper={FIX / 'whisper.json'}",
                   "--out", str(tmp_path / "o")]) == 64


def test_bad_transcript_spec_exits_64(tmp_path, speech_wav):
    assert g.main(["--audio", str(speech_wav), "--transcript", "whisper",
                   "--out", str(tmp_path / "o")]) == 64
    assert g.main(["--audio", str(speech_wav), "--transcript", "whisper=/no/such.json",
                   "--out", str(tmp_path / "o")]) == 64
    assert g.main(["--audio", str(speech_wav), "--transcript", "bad name=x.json",
                   "--out", str(tmp_path / "o")]) == 64


def test_duplicate_transcript_name_exits_64(tmp_path, speech_wav):
    spec = f"whisper={FIX / 'whisper.json'}"
    assert g.main(["--audio", str(speech_wav), "--transcript", spec, "--transcript", spec,
                   "--out", str(tmp_path / "o")]) == 64


def test_unknown_primary_exits_64(tmp_path, speech_wav):
    assert g.main(["--audio", str(speech_wav),
                   "--transcript", f"whisper={FIX / 'whisper.json'}",
                   "--primary", "kts", "--out", str(tmp_path / "o")]) == 64


def test_invalid_vad_choice_exits_64(tmp_path, speech_wav):
    assert g.main(["--audio", str(speech_wav),
                   "--transcript", f"whisper={FIX / 'whisper.json'}",
                   "--vad", "bogus", "--out", str(tmp_path / "o")]) == 64


def test_out_is_file_exits_64(tmp_path, speech_wav):
    f = tmp_path / "file.bin"
    f.write_text("x", encoding="utf-8")
    assert g.main(["--audio", str(speech_wav),
                   "--transcript", f"whisper={FIX / 'whisper.json'}",
                   "--out", str(f)]) == 64


# ---------------------------------------------------------------------------
# LID offline: подмена только внешнего провайдера, без импорта ML и сети
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("available", [False, True])
def test_whisper_lid_uses_only_local_weights(tmp_path, speech_wav, monkeypatch, available):
    constructors, transcriptions = [], []

    class LocalWhisper:
        def __init__(self, model_name, **kwargs):
            constructors.append((model_name, kwargs))
            if not available:
                raise FileNotFoundError("локальная модель отсутствует")

        def transcribe(self, sample, **kwargs):
            transcriptions.append(kwargs)
            return [], SimpleNamespace(language="ru", language_probability=0.9876)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=LocalWhisper))
    code, report = run_gate_cli(tmp_path, speech_wav, {"whisper": WHISPER_LIKE},
                                "--vad", "energy", "--lid", "whisper", "--stt-model", "local-model")
    assert constructors == [("local-model", {"device": "cpu", "compute_type": "int8",
                                            "local_files_only": True})]
    assert len(transcriptions) == int(available)
    assert report["language_id"] == (
        {"language": "ru", "probability": 0.988, "method": "whisper"} if available else
        {"language": None, "probability": None, "method": "whisper_unavailable"})
    assert report["config"]["thresholds"]["lid_offline"] is True
    assert code == 0


# ---------------------------------------------------------------------------
# ML-путь: помечено skip, в тестах не гоняем
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="ML: faster-whisper WhisperModel (модель/сеть)")
def test_lid_whisper_detects_russian():
    x = tone(5 * SR)
    assert g.detect_language(x, SR, "large-v3")["language"] == "ru"


@pytest.mark.skip(reason="ML: faster-whisper Silero VAD (onnx-модель)")
def test_vad_silero_activity():
    x = np.concatenate([tone(3 * SR), np.zeros(SR, np.float32), tone(2 * SR)])
    assert g.vad_activity(x, SR)["active_s"] > 0


# ---------------------------------------------------------------------------
# LOW_INFORMATION: подтверждения, пороги и пустой текст
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("short_count,total,severity", [
    (1, 5, None),
    (3, 10, "warn"),
    (4, 10, "warn"),
    (5, 10, "block"),
    (6, 10, "block"),
    (4, 4, None),
    (4, 5, "block"),
])
def test_low_information_fraction_thresholds(short_count, total, severity):
    segs = [{"start": i, "end": i + 1.0,
             "text": "шум" if i < short_count else "полезная реплика"}
            for i in range(total)]
    reasons = g.build_reasons(**_facts(segments=segs))
    low = [r for r in reasons if r["code"] == "LOW_INFORMATION"]
    assert [r["severity"] for r in low] == ([] if severity is None else [severity])
    if low:
        assert low[0]["intervals"] == [{"start": 0.0, "end": float(short_count)}]


@pytest.mark.parametrize("confirmation", [
    "Да.", "Нет.", "Спасибо.", "Понял!", "ПОНЯЛА.", "Хорошо.",
    "ОК.", "Ага.", "Угу.", "Верно.", "Точно.",
])
def test_confirmation_in_short_segments_prevents_low_information(confirmation):
    texts = ["шум"] * 4 + [confirmation]
    segs = [{"start": i, "end": i + 1.0, "text": text} for i, text in enumerate(texts)]
    assert g.build_reasons(**_facts(segments=segs)) == []


@pytest.mark.parametrize("short_duration,severity", [(0.49, None), (0.5, "block")])
def test_low_information_considers_only_segments_at_least_half_second(short_duration, severity):
    segs = [{"start": i, "end": i + 1.0, "text": "полезная реплика"} for i in range(5)]
    segs += [{"start": i, "end": i + short_duration, "text": "шум"} for i in range(5, 10)]
    low = [r for r in g.build_reasons(**_facts(segments=segs)) if r["code"] == "LOW_INFORMATION"]
    assert [r["severity"] for r in low] == ([] if severity is None else [severity])


@pytest.mark.parametrize("empty_count,total,severity", [
    (1, 2, None),        # единичная пауза не считается деградацией
    (2, 5, "warn"),
    (3, 10, None),       # для пустых при подтверждениях доля строго > 0.30
    (4, 10, "warn"),
    (5, 10, "block"),
    (9, 10, "block"),
    (2, 3, "block"),     # пустые проверяются и при числе сегментов < INFO_MIN_SEGMENTS
])
@pytest.mark.parametrize("duration", [0.1, 1.0])
def test_empty_segments_are_degradation_even_with_confirmations(empty_count, total, severity, duration):
    segs = [{"start": i, "end": i + duration, "text": "" if i < empty_count else "Да."}
            for i in range(total)]
    reasons = g.build_reasons(**_facts(segments=segs))
    assert [(r["code"], r["severity"]) for r in reasons] == (
        [] if severity is None else [("LOW_INFORMATION", severity)])
    if reasons:
        assert reasons[0]["intervals"] == g.to_interval_dicts(
            [(s["start"], s["end"]) for s in segs[:empty_count]])


@pytest.mark.parametrize("empty_text", ["", "   ", None, " ... "])
def test_one_confirmation_and_nine_empty_segments_block_cli(tmp_path, empty_text):
    wav = make_wav(tmp_path / "empty.wav", [(0, 10)], 10.0)
    segs = [{"start": i, "end": i + 1.0, "text": "Да." if i == 0 else empty_text}
            for i in range(10)]
    code, report = run_gate_cli(tmp_path, wav, {"whisper": segs},
                                "--vad", "energy", "--lid", "declared")
    assert code == 2
    assert report["verdict"] == "failed"
    assert report["coverage_union"]["coverage_union"] == 1.0
    assert [(r["code"], r["severity"]) for r in report["reasons"]] == [("LOW_INFORMATION", "block")]
    assert report["reasons"][0]["intervals"] == [{"start": 1.0, "end": 10.0}]


def test_low_information_thresholds_are_reported(tmp_path, speech_wav):
    _, report = run_gate_cli(tmp_path, speech_wav, {"whisper": WHISPER_LIKE},
                             "--vad", "energy", "--lid", "declared")
    thresholds = report["config"]["thresholds"]
    assert thresholds["info_degenerate_warn"] == 0.30
    assert thresholds["info_degenerate_block"] == 0.50
    assert thresholds["info_min_segments"] == 5
    assert thresholds["info_min_segment_s"] == 0.5
    assert thresholds["info_min_words"] == 2


@pytest.mark.parametrize("repeats", [1, 2])
def test_short_confirmation_dialog_is_not_low_information(tmp_path, speech_wav, repeats):
    texts = ["Да.", "Нет.", "Спасибо."] * repeats
    segs = [{"start": i, "end": i + 1.0, "text": text} for i, text in enumerate(texts)]
    _, report = run_gate_cli(tmp_path, speech_wav, {"whisper": segs},
                             "--vad", "energy", "--lid", "declared")
    assert "LOW_INFORMATION" not in {r["code"] for r in report["reasons"]}


@pytest.mark.parametrize("empty_segments", [
    [],
    [{"start": 0.0, "end": 0.1, "text": ""}],
    [{"start": 0.0, "end": 0.49, "text": " ... "}],
    [{"start": 0.0, "end": 0.2, "text": None}],
])
def test_all_empty_text_blocks_regardless_of_duration_or_segment_count(empty_segments):
    reasons = g.build_reasons(**_facts(segments=empty_segments))
    assert [(r["code"], r["severity"]) for r in reasons] == [("LOW_INFORMATION", "block")]
    assert g.verdict_of(reasons) == "failed"


def test_short_nonempty_segment_prevents_all_empty_block():
    segs = [{"start": 0.0, "end": 0.2, "text": "Спасибо."},
            {"start": 1.0, "end": 2.0, "text": ""}]
    assert g.build_reasons(**_facts(segments=segs)) == []


def test_empty_segment_list_blocks_without_audio_activity(tmp_path):
    wav = make_wav(tmp_path / "silence.wav", [], 1.0)
    code, report = run_gate_cli(tmp_path, wav, {"whisper": []},
                                "--vad", "energy", "--lid", "declared")
    assert code == 2
    assert report["verdict"] == "failed"
    assert report["coverage_union"]["coverage_union"] is None
    assert {r["code"]: r["severity"] for r in report["reasons"]}["LOW_INFORMATION"] == "block"


@pytest.mark.parametrize("duration", [0.1, 0.49, 6.0])
def test_empty_transcript_is_blocked_even_with_one_segment(tmp_path, speech_wav, duration):
    """Покрытие интервалами без текста — не «ok»: один пустой сегмент = block."""
    _, r = run_gate_cli(tmp_path, speech_wav,
                        {"whisper": [{"start": 0.0, "end": duration, "text": "   "}]},
                        "--vad", "energy", "--lid", "declared")
    codes = {x["code"] for x in r["reasons"]}
    assert "LOW_INFORMATION" in codes
    assert r["verdict"] == "failed"
    reason = [x for x in r["reasons"] if x["code"] == "LOW_INFORMATION"][0]
    assert reason["severity"] == "block"
