#!/usr/bin/env python3
"""Tests for asr_repair.py (plan/run/merge) and engines/whisper_ct2.py.

No network, no ML: the engine adapter is exercised through its pure document
builder, and `run` is driven by a fake engine script that only writes JSON.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asr_repair as rep  # noqa: E402

FIXTURES = ROOT / "fixtures"


def load_whisper_doc() -> dict:
    return {"schema": "asr.segments/1",
            "audio": {"path": "/x/audio.wav", "sha256": "a" * 64, "duration_s": 178.25,
                      "sr": 16000, "channels": 1},
            "engine": {"name": "whisper", "model": "large-v3", "version": "ct2",
                       "params": {}, "level_sensitive": True},
            "segments": json.loads((FIXTURES / "whisper.json").read_text(encoding="utf-8"))["segments"],
            "meta": {"vad": "silero", "window_s": None}}


def kts_segments() -> list[dict]:
    return json.loads((FIXTURES / "kts.json").read_text(encoding="utf-8"))["segments"]


def seg(start, end, text, **kw):
    row = {"id": kw.get("id", ""), "start": start, "end": end, "text": text,
           "speaker": None, "conf": None,
           "source": kw.get("source", "primary"), "flags": list(kw.get("flags", []))}
    return row


def additions_doc(segments, holes, attempt=1):
    return {"schema": "asr.segments/1",
            "audio": {"path": "/x/audio.wav", "sha256": "a" * 64, "duration_s": 178.25,
                      "sr": 16000, "channels": 1},
            "engine": {"name": "whisper", "model": "large-v3", "version": "repair",
                       "params": {}, "level_sensitive": True},
            "segments": segments,
            "meta": {"vad": "off", "window_s": None,
                     "repair": {"attempt": attempt,
                                "intervals": [{"id": h[0], "start": h[1], "end": h[2]} for h in holes]}}}


def base_doc(segments):
    return {"schema": "asr.segments/1",
            "audio": {"path": "/x/audio.wav", "sha256": "a" * 64, "duration_s": 100.0,
                      "sr": 16000, "channels": 1},
            "engine": {"name": "whisper", "model": "large-v3", "version": "ct2",
                       "params": {}, "level_sensitive": True},
            "segments": segments,
            "meta": {"vad": "off", "window_s": None}}


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def trust_doc(uncovered, duration=100.0):
    return {"schema": "asr.trust/2", "verdict": "partial", "primary": "whisper",
            "audio_metrics": {"duration_s": duration},
            "uncovered_all": [{"start": a, "end": b} for a, b in uncovered]}


def test_plan_pads_merges_and_drops_short_holes():
    plan = rep.build_plan(trust_doc([(0.0, 3.0), (3.9, 5.2), (0.2, 0.7), (50.0, 52.0)]))
    assert plan["schema"] == "asr.repair.plan/1"
    assert plan["attempt"] == 1
    assert [iv["id"] for iv in plan["intervals"]] == ["r1", "r2"]
    first, second = plan["intervals"]
    assert (first["start"], first["end"]) == (0.0, 5.5)          # 0.3 pad + 0.3 gap merge
    assert (second["start"], second["end"]) == (49.7, 52.3)      # 52.0 + 0.3 pad
    assert all(iv["pad_s"] == 0.3 and iv["reason"] == "gap" for iv in plan["intervals"])
    assert all(iv["engines"] == ["whisper"] for iv in plan["intervals"])
    assert first["decode_s"] == 5.8 and second["decode_s"] == 2.9
    assert plan["total_decode_s"] == 8.7
    assert plan["budget_s"] == 35.0
    assert plan["within_budget"] is True
    assert plan["stop_rule"] == {"min_gain_ratio": 0.25, "max_attempts": 2}


def test_plan_drops_holes_below_one_second_and_intervals_below_1_5():
    # 0.9 s hole is not a hole; the 1.0 s hole at the very end pads to 1.3 s < 1.5 s
    plan = rep.build_plan(trust_doc([(10.0, 10.9), (99.0, 100.0)]))
    assert plan["intervals"] == []
    assert plan["total_decode_s"] == 0.0
    assert plan["within_budget"] is True


def test_plan_marks_over_budget():
    plan = rep.build_plan(trust_doc([(0.0, 5.0)], duration=10.0))
    assert plan["budget_s"] == 3.5
    assert plan["total_decode_s"] == 5.6      # 0-5.0 hole + 0.3 pad + 0.3 cut tail
    assert plan["within_budget"] is False


def test_plan_ignores_route_intervals_but_takes_budget():
    route = {"schema": "asr.route/1",
             "steps": [{"id": "repair_decode", "kind": "repair", "because": "SPEECH_GAP",
                        "params": {"intervals": [{"start": 0.0, "end": 40.0}]}}],
             "budget": {"attempts_max": 3, "seconds_max": 12.0}}
    plan = rep.build_plan(trust_doc([(50.0, 52.0)]), route)
    assert len(plan["intervals"]) == 1
    assert plan["intervals"][0]["start"] == 49.7          # trust hole, not route's 0-40
    assert plan["budget_s"] == 12.0
    assert plan["stop_rule"]["max_attempts"] == 3


def test_plan_clamps_to_duration_and_requires_duration():
    plan = rep.build_plan(trust_doc([(97.5, 100.0)]))
    assert plan["intervals"][0]["end"] == 100.0
    with pytest.raises(rep.RepairError) as exc:
        rep.build_plan({"schema": "asr.trust/2", "uncovered_all": []})
    assert "PLAN_NO_DURATION" in str(exc.value)


def test_cli_bad_arguments_exit_64():
    with pytest.raises(SystemExit) as exc:
        rep.main(["plan", "--trust", "x.json"])
    assert exc.value.code == 64


def test_plan_cli_writes_plan_and_alias(tmp_path):
    trust = tmp_path / "trust.json"
    trust.write_text(json.dumps(trust_doc([(50.0, 52.0)])), encoding="utf-8")
    out = tmp_path / "30_repair"
    assert rep.main(["plan", "--trust", str(trust), "--out", str(out)]) == 0
    doc = json.loads((out / "repair_plan.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "asr.repair.plan/1" and len(doc["intervals"]) == 1
    assert json.loads((out / "plan.json").read_text(encoding="utf-8")) == doc


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def test_merge_clips_additions_at_hole_boundaries():
    base = base_doc([seg(0.0, 5.0, "начало"), seg(10.0, 15.0, "конец")])
    add = additions_doc([seg(4.8, 7.2, "вставка раз", source="repair:whisper:r1"),
                         seg(8.0, 10.2, "вставка два", source="repair:whisper:r1")],
                        [("r1", 5.0, 10.0)])
    merged = rep.merge_documents(base, add)
    spans = [(s["start"], s["end"]) for s in merged["segments"]]
    assert spans == [(0.0, 5.0), (5.0, 7.2), (8.0, 10.0), (10.0, 15.0)]
    assert merged["meta"]["repair"]["coverage_before_s"] == 10.0
    assert merged["meta"]["repair"]["coverage_after_s"] == 14.2   # 7.2-8.0 stays uncovered
    assert merged["meta"]["repair"]["added"] == 2


def test_merge_removes_overlap_and_keeps_monotonic():
    base = base_doc([seg(0.0, 5.0, "начало"), seg(10.0, 15.0, "конец")])
    add = additions_doc([seg(7.0, 10.0, "второй кусок", source="repair:whisper:r1"),
                         seg(5.0, 8.0, "первый кусок", source="repair:whisper:r1")],
                        [("r1", 5.0, 10.0)])
    merged = rep.merge_documents(base, add)
    spans = [(s["start"], s["end"]) for s in merged["segments"]]
    assert spans == [(0.0, 5.0), (5.0, 8.0), (8.0, 10.0), (10.0, 15.0)]
    starts = [s["start"] for s in merged["segments"]]
    assert starts == sorted(starts)
    assert all(s["start"] < s["end"] for s in merged["segments"])
    for a, b in zip(merged["segments"], merged["segments"][1:]):
        assert a["end"] - b["start"] <= rep.MAX_OVERLAP_S


def test_merge_rejects_duplicate_that_adds_no_coverage():
    base = base_doc([seg(0.0, 10.0, "сплошной текст без дыры")])
    add = additions_doc([seg(2.0, 8.0, "тот же участок заново", source="repair:whisper:r1")],
                        [("r1", 2.0, 8.0)])
    with pytest.raises(rep.MergeRejected) as exc:
        rep.merge_documents(base, add)
    assert any(v.startswith("M4") for v in exc.value.violations)


def test_merge_rejects_addition_outside_planned_interval():
    base = base_doc([seg(0.0, 5.0, "начало"), seg(10.0, 15.0, "конец")])
    add = additions_doc([seg(20.0, 25.0, "чужая вставка", source="repair:whisper:r1")],
                        [("r1", 5.0, 10.0)])
    with pytest.raises(rep.MergeRejected) as exc:
        rep.merge_documents(base, add)
    assert any(v.startswith("M2") for v in exc.value.violations)


def test_merge_rejects_repeated_ngram_within_two_seconds():
    phrase = "это очень длинная фраза которая повторяется дважды подряд"
    base = base_doc([seg(0.0, 5.0, "начало"), seg(10.0, 15.0, "конец")])
    add = additions_doc([seg(5.0, 7.0, phrase, source="repair:whisper:r1"),
                         seg(6.9, 9.5, phrase, source="repair:whisper:r1")],
                        [("r1", 5.0, 10.0)])
    with pytest.raises(rep.MergeRejected) as exc:
        rep.merge_documents(base, add)
    assert any(v.startswith("M3") for v in exc.value.violations)


def test_merge_cli_returns_base_and_exit_2_on_rejection(tmp_path, capsys):
    base = base_doc([seg(0.0, 10.0, "сплошной текст без дыры")])
    add = additions_doc([seg(2.0, 8.0, "тот же участок заново", source="repair:whisper:r1")],
                        [("r1", 2.0, 8.0)])
    base_path, add_path, out = tmp_path / "base.json", tmp_path / "add.json", tmp_path / "out.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    add_path.write_text(json.dumps(add), encoding="utf-8")
    original_base = base_path.read_bytes()
    assert rep.main(["merge", "--base", str(base_path), "--additions", str(add_path),
                     "--out", str(out)]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "MERGE_REJECTED"
    assert payload["ok"] is False
    assert json.loads(out.read_text(encoding="utf-8")) == base
    assert base_path.read_bytes() == original_base


def test_merge_keeps_ids_unique_when_additions_reuse_base_ids():
    base = base_doc([seg(0.0, 5.0, "начало", id="s0001"), seg(10.0, 15.0, "конец", id="s0002")])
    add = additions_doc([seg(5.0, 7.5, "вставка", id="s0001", source="repair:whisper:r1")],
                        [("r1", 5.0, 10.0)])
    merged = rep.merge_documents(base, add)
    ids = [s["id"] for s in merged["segments"]]
    assert len(ids) == len(set(ids)), ids


def test_clip_splice_on_real_fixtures_grows_union_without_overlaps():
    base = load_whisper_doc()
    before = rep.union_length(base["segments"])
    holes = [(f"r{i + 1}", a, b)
             for i, (a, b) in enumerate(rep.merge_intervals(
                 [(s["end"], t["start"]) for s, t in zip(base["segments"], base["segments"][1:])
                  if t["start"] - s["end"] >= rep.MIN_HOLE_S]))]
    assert len(holes) == 3, holes
    # what `run` would emit: only KTS segments that actually overlap a planned hole
    add_segs = []
    for s in kts_segments():
        best = max(((min(s["end"], h[2]) - max(s["start"], h[1]), h) for h in holes),
                   key=lambda x: x[0])
        if best[0] <= 0:
            continue
        add_segs.append({"id": "", "start": s["start"], "end": s["end"], "text": s["text"],
                         "speaker": None, "conf": None,
                         "source": f"repair:whisper:{best[1][0]}", "flags": ["clipped"]})
    assert len(add_segs) == 3
    add = additions_doc(add_segs, holes)
    merged = rep.merge_documents(base, add)
    after = rep.union_length(merged["segments"])
    assert after > before
    # 3 big holes filled; the sub-second 104.57-105.11 gap is not planned, so 169.55
    assert after == pytest.approx(before + 8.66, abs=0.05)
    assert merged["meta"]["repair"]["added"] == 3
    assert merged["meta"]["repair"]["gain_ratio"] == pytest.approx(1.0, abs=0.01)
    for a, b in zip(merged["segments"], merged["segments"][1:]):
        assert a["end"] - b["start"] <= 0.2
    filler = [s for s in merged["segments"] if s["source"].startswith("repair:")]
    assert [round(s["start"], 2) for s in filler] == [95.98, 100.89, 165.05]
    assert merged["meta"]["repair"]["trimmed"] == 0   # base segments already stop at the holes
    assert all(s["text"] for s in merged["segments"])


def test_merge_without_holes_falls_back_to_additions_union():
    base = base_doc([seg(0.0, 5.0, "начало"), seg(10.0, 15.0, "конец")])
    add = {"schema": "asr.segments/1",
           "segments": [seg(5.0, 10.0, "вставка", source="repair:whisper:r1")],
           "meta": {"vad": "off", "window_s": None}}
    merged = rep.merge_documents(base, add)
    assert [(s["start"], s["end"]) for s in merged["segments"]] == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]


# --------------------------------------------------------------------------- #
# run (fake engine, real ffmpeg cut)
# --------------------------------------------------------------------------- #
FAKE_ENGINE = '''
import argparse, json
from pathlib import Path
import soundfile as sf

p = argparse.ArgumentParser()
p.add_argument("--audio"); p.add_argument("--out"); p.add_argument("--language")
a = p.parse_args()
info = sf.info(a.audio)
doc = {"schema": "asr.segments/1",
       "audio": {"path": a.audio, "sha256": "0" * 64, "duration_s": info.duration,
                 "sr": 16000, "channels": 1},
       "engine": {"name": "fake", "model": "none", "version": "test",
                  "params": {}, "level_sensitive": False},
       "segments": [
           {"id": "s1", "start": 0.05, "end": 0.4, "text": "тест раз", "speaker": None,
            "conf": None, "source": "primary", "flags": []},
           {"id": "s2", "start": 1.0, "end": 1.8, "text": "тест два", "speaker": None,
            "conf": None, "source": "primary", "flags": []},
           {"id": "s3", "start": 2.0, "end": 2.2, "text": "   ", "speaker": None,
            "conf": None, "source": "primary", "flags": []}],
       "meta": {"vad": "off", "window_s": None}}
_out = Path(a.out)
if _out.suffix == ".json":
    _out.parent.mkdir(parents=True, exist_ok=True)
    _target = _out
else:
    _out.mkdir(parents=True, exist_ok=True)
    _target = _out / "whisper.segments.json"
_target.write_text(json.dumps(doc), encoding="utf-8")
print("fake engine done")
'''


def make_wav(path: Path, seconds: float = 6.0, sr: int = 16000) -> Path:
    import numpy as np
    import soundfile as sf

    t = np.arange(int(seconds * sr)) / sr
    x = (0.1 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(str(path), x, sr, subtype="PCM_16")
    return path


@pytest.fixture
def repair_run_inputs(tmp_path):
    audio = make_wav(tmp_path / "audio.wav")
    counter = tmp_path / "engine_calls.txt"
    counter.write_text("0", encoding="utf-8")
    engine_py = tmp_path / "fake_engine.py"
    engine_py.write_text(
        "from pathlib import Path\n"
        f"counter = Path({str(counter)!r})\n"
        "counter.write_text(str(int(counter.read_text()) + 1))\n" + FAKE_ENGINE,
        encoding="utf-8",
    )
    engines = tmp_path / "engines.json"
    engines.write_text(json.dumps({"fake": {
        "cmd": [sys.executable, str(engine_py), "--audio", "{audio}",
                "--out", "{out}", "--language", "{language}"],
        "timeout_s": 120}}), encoding="utf-8")
    plan = rep.build_plan(
        trust_doc([(1.0, 3.0)], duration=6.0),
        {"budget": {"seconds_max": 0.1},
         "steps": [{"kind": "repair", "params": {"engines": ["fake"]}}]},
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    out = tmp_path / "30_repair"
    argv = ["run", "--audio", str(audio), "--plan", str(plan_path),
            "--engines", str(engines), "--out", str(out)]
    return argv, plan, plan_path, out, counter


def test_run_blocks_over_budget_before_decoding(repair_run_inputs, capsys):
    argv, plan, plan_path, out, counter = repair_run_inputs
    original_plan = plan_path.read_bytes()
    assert plan["within_budget"] is False
    assert plan["total_decode_s"] > plan["budget_s"]

    rc = rep.main(argv)

    assert counter.read_text(encoding="utf-8") == "0"
    assert rc == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "blocked", "code": "REPAIR_BUDGET_EXCEEDED",
                       "planned_decode_s": plan["total_decode_s"],
                       "budget_s": plan["budget_s"]}
    assert not out.exists()
    assert plan_path.read_bytes() == original_plan


def test_run_force_budget_explicitly_allows_decoding(repair_run_inputs, capsys):
    argv, plan, plan_path, out, counter = repair_run_inputs
    original_plan = plan_path.read_bytes()
    assert plan["within_budget"] is False

    assert rep.main(argv + ["--force-budget"]) == 0

    assert counter.read_text(encoding="utf-8") == "1"
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    attempt = json.loads((out / "attempt-1.json").read_text(encoding="utf-8"))
    assert attempt["status"] == "ok"
    assert attempt["total_decode_s"] > attempt["budget_s"]
    assert (out / "additions.json").is_file()
    assert plan_path.read_bytes() == original_plan


@pytest.mark.parametrize("budget_s", [2.3, 2.5], ids=["at_budget", "under_budget"])
def test_run_cuts_calls_engine_and_clips_segments(tmp_path, budget_s):
    audio = make_wav(tmp_path / "audio.wav")
    engine_py = tmp_path / "fake_engine.py"
    engine_py.write_text(FAKE_ENGINE, encoding="utf-8")
    engines = tmp_path / "engines.json"
    engines.write_text(json.dumps({"fake": {
        "cmd": [sys.executable, str(engine_py), "--audio", "{audio}",
                "--out", "{out}", "--language", "{language}"],
        "model": "none", "level_sensitive": False, "timeout_s": 120}}), encoding="utf-8")
    plan = {"schema": "asr.repair.plan/1", "attempt": 1,
            "intervals": [{"id": "r1", "start": 1.0, "end": 3.0, "pad_s": 0.3,
                           "reason": "gap", "engines": ["fake"], "decode_s": 2.3}],
            "total_decode_s": 2.3, "budget_s": budget_s, "within_budget": True,
            "stop_rule": {"min_gain_ratio": 0.25, "max_attempts": 2}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    out = tmp_path / "30_repair"

    rc = rep.main(["run", "--audio", str(audio), "--plan", str(plan_path),
                   "--engines", str(engines), "--out", str(out)])
    assert rc == 0

    attempt = json.loads((out / "attempt-1.json").read_text(encoding="utf-8"))
    assert attempt["schema"] == "asr.repair.attempt/1"
    assert attempt["status"] == "ok"
    assert attempt["intervals"][0]["engine"] == "fake"
    assert attempt["intervals"][0]["offset_s"] == pytest.approx(0.85, abs=1e-3)
    assert attempt["segments_n"] == 2          # empty text dropped
    assert Path(attempt["intervals"][0]["cut"]).is_file()

    doc = json.loads((out / "attempt-1.segments.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "asr.segments/1"
    spans = [(s["start"], s["end"]) for s in doc["segments"]]
    # local 0.05 -> global 0.90, clipped to the hole at 1.0
    assert spans[0] == (1.0, 1.25)
    assert spans[1] == (1.85, 2.65)
    for s in doc["segments"]:
        assert s["source"] == "repair:fake:r1"
        assert "clipped" in s["flags"]
        assert 1.0 <= s["start"] < s["end"] <= 3.0
    assert [s["id"] for s in doc["segments"]] == ["s0001", "s0002"]
    assert doc["meta"]["repair"]["intervals"] == [{"id": "r1", "start": 1.0, "end": 3.0}]
    # pipeline alias written next to the attempt files
    assert json.loads((out / "additions.json").read_text(encoding="utf-8")) == doc

    # the attempt document feeds merge directly
    merged = rep.merge_documents(
        base_doc([seg(0.0, 1.0, "было"), seg(3.0, 6.0, "стало")]), doc)
    assert rep.union_length(merged["segments"]) == pytest.approx(5.05, abs=1e-6)


def test_run_engine_failure_preserves_base_and_exit_2(repair_run_inputs, tmp_path, capsys):
    argv, _, _, out, counter = repair_run_inputs
    engine_py = tmp_path / "fake_engine.py"
    # Even a produced segments file must not be accepted after a nonzero exit.
    engine_py.write_text(engine_py.read_text(encoding="utf-8") + "\nraise SystemExit(7)\n",
                         encoding="utf-8")
    base = base_doc([seg(0.0, 0.7, "начало"), seg(3.3, 6.0, "конец")])
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    original_base = base_path.read_bytes()

    assert rep.main(argv + ["--force-budget"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed" and payload["ok"] is False
    assert counter.read_text(encoding="utf-8") == "1"
    assert base_path.read_bytes() == original_base
    attempt = json.loads((out / "attempt-1.json").read_text(encoding="utf-8"))
    assert attempt["status"] == "failed"
    assert attempt["intervals"][0]["rc"] == 7
    assert "ENGINE_FAILED" in attempt["intervals"][0]["error"]
    additions = out / "additions.json"
    assert json.loads(additions.read_text(encoding="utf-8"))["segments"] == []

    merged = out / "merged.json"
    assert rep.main(["merge", "--base", str(base_path), "--additions", str(additions),
                     "--out", str(merged)]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == "MERGE_REJECTED"
    assert json.loads(merged.read_text(encoding="utf-8")) == base
    assert base_path.read_bytes() == original_base


def test_run_reports_unknown_engine(tmp_path):
    audio = make_wav(tmp_path / "audio.wav")
    plan = {"schema": "asr.repair.plan/1", "attempt": 1,
            "intervals": [{"id": "r1", "start": 1.0, "end": 2.0, "engines": ["nope"]}]}
    with pytest.raises(rep.RepairError) as exc:
        rep.run_attempt(audio, plan, {}, tmp_path / "out")
    assert exc.value.code == "ENGINE_UNKNOWN"


# --------------------------------------------------------------------------- #
# whisper_ct2 adapter (no model load)
# --------------------------------------------------------------------------- #
def load_adapter():
    path = ROOT / "engines" / "whisper_ct2.py"
    spec = importlib.util.spec_from_file_location("whisper_ct2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_adapter_document_contract(tmp_path):
    adapter = load_adapter()
    wav = make_wav(tmp_path / "cut.wav", seconds=4.0)
    x, sr = adapter.load_audio(wav)
    assert sr == 16000 and x.ndim == 1
    rows = [{"start": 0.0, "end": 1.5, "text": " раз ", "conf": 0.9},
            {"start": 1.2, "end": 2.0, "text": "два", "conf": 0.5},   # starts earlier: monotonic fix
            {"start": 2.5, "end": 2.5, "text": "пусто по времени", "conf": None},
            {"start": 3.0, "end": 9.0, "text": "  ", "conf": None}]   # empty text dropped
    doc = adapter.build_doc(wav, x, sr, rows, "large-v3", "ru", {"language_probability": 0.97})
    assert doc["schema"] == "asr.segments/1"
    assert doc["engine"]["name"] == "whisper"
    assert doc["engine"]["model"] == "large-v3"
    assert doc["engine"]["version"] == "ct2"
    assert doc["engine"]["level_sensitive"] is True
    assert doc["engine"]["params"]["local_files_only"] is True
    assert doc["audio"]["duration_s"] == 4.0 and doc["audio"]["channels"] == 1
    assert len(doc["audio"]["sha256"]) == 64
    starts = [s["start"] for s in doc["segments"]]
    assert starts == sorted(starts)
    assert all(s["start"] < s["end"] for s in doc["segments"])
    assert [s["id"] for s in doc["segments"]] == [f"s{i:04d}" for i in range(1, len(doc["segments"]) + 1)]
    assert all(s["text"] for s in doc["segments"])
    assert doc["meta"]["language"] == "ru"


def test_adapter_decode_params_match_contract():
    adapter = load_adapter()
    assert adapter.DEFAULT_MODEL == "large-v3"
    assert adapter.DECODE_PARAMS["beam_size"] == 5
    assert adapter.DECODE_PARAMS["vad_filter"] is False
    assert adapter.DECODE_PARAMS["temperature"] == 0.0
    assert adapter.DECODE_PARAMS["condition_on_previous_text"] is False
    args = adapter.parse_args(["--audio", "a.wav", "--out", "o.json"])
    assert args.language == "ru"
