"""Offline regressions for the medoid ensemble and x20 adapter (no ML)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asr_ensemble as E
from engines import x20_engine as X


def segment(start, end, text):
    return {"start": start, "end": end, "text": text}


def run_ensemble(tmp_path, monkeypatch, engines, *, window=20):
    argv = ["asr_ensemble.py", "--out", str(tmp_path / "out"), "--window", str(window)]
    for i, (name, segments) in enumerate(engines.items()):
        path = tmp_path / f"input-{i}.json"
        path.write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")
        argv += ["--segments", f"{name}={path}"]
    monkeypatch.setattr(sys, "argv", argv)
    assert E.main() == 0
    doc = json.loads((tmp_path / "out" / "segments.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    return doc, report


def test_mean_confidence_rounds_raw_scores_only_once(tmp_path, monkeypatch):
    doc, report = run_ensemble(tmp_path, monkeypatch, {
        "first": [segment(0, 20, "общий"), segment(20, 40, "общий")],
        "second": [segment(0, 20, "общий " + " ".join(f"x{i}" for i in range(12))),
                   segment(20, 40, "общий другой третий")],
    })
    assert [s["conf"] for s in doc["segments"]] == [0.143, 0.5]
    assert report["mean_confidence"] == round((1 / 7 + 1 / 2) / 2, 3) == 0.321


@pytest.mark.parametrize("start,end,expected", [
    (15, 25, ["пограничный", ""]),  # Equal overlap: keep the earlier window.
    (19, 25, ["", "пограничный"]),  # A small leading overlap must not copy the text.
    (0, 40, ["пограничный", ""]),   # Both pass the overlap threshold, still only one owner.
    (20, 40, ["", "пограничный"]),
    (20, 20, ["", ""]),
])
def test_window_text_has_exactly_one_owner(start, end, expected):
    segments = [segment(start, end, "пограничный")]
    assert [E.window_text(segments, 0, 20), E.window_text(segments, 20, 40)] == expected


def test_crossing_segment_is_not_duplicated_in_ensemble_output(tmp_path, monkeypatch):
    doc, report = run_ensemble(tmp_path, monkeypatch, {
        "first": [segment(15, 25, "один сегмент")],
        "second": [segment(15, 25, "один сегмент")],
    })
    assert [(s["start"], s["end"], s["text"]) for s in doc["segments"]] == [(0, 20, "один сегмент")]
    assert report["segments"] == 1


def test_short_final_window_uses_the_same_grid(tmp_path, monkeypatch):
    doc, _ = run_ensemble(tmp_path, monkeypatch, {
        "first": [segment(15, 21, "на границе"), segment(20, 21, "хвост")],
        "second": [segment(15, 21, "на границе"), segment(20, 21, "хвост")],
    })
    assert [s["text"] for s in doc["segments"]] == ["на границе", "хвост"]
    assert doc["segments"][-1]["end"] == 21


def test_maximum_overlap_fallback_with_subsecond_windows(tmp_path, monkeypatch):
    doc, _ = run_ensemble(tmp_path, monkeypatch, {
        "first": [segment(0.1, 2.1, "длинный сегмент")],
        "second": [segment(0.1, 2.1, "длинный сегмент")],
    }, window=0.5)
    # No window contains >=50% or >=2 s; the earliest maximum overlap wins.
    assert [(s["start"], s["end"], s["text"]) for s in doc["segments"]] == [(0.5, 1, "длинный сегмент")]


@pytest.mark.parametrize("empty", [[], [segment(0, 20, "")], [segment(0, 20, " \n ")], [segment(0, 20, None)]])
def test_single_source_has_no_agreement_confidence(tmp_path, monkeypatch, empty):
    doc, report = run_ensemble(tmp_path, monkeypatch, {
        "empty_first": empty,
        "only": [segment(0, 20, "единственный источник")],
        "empty_last": empty,
    })
    assert doc["segments"][0]["source"] == "only"
    assert doc["segments"][0]["conf"] == 0.0
    assert doc["segments"][0]["flags"] == ["single_source"]
    assert report["windows"][0]["scores"] == {"only": 0.0}
    assert report["windows"][0]["single_source"] is True
    assert report["windows"][0]["empty"] == ["empty_first", "empty_last"]
    assert report["mean_confidence"] == 0.0


def test_medoid_wins_by_mean_token_dice(tmp_path, monkeypatch):
    doc, report = run_ensemble(tmp_path, monkeypatch, {
        "left": [segment(0, 20, "альфа бета")],
        "middle": [segment(0, 20, "альфа бета гамма")],
        "right": [segment(0, 20, "бета гамма")],
    })
    assert doc["segments"][0]["source"] == "middle"
    assert doc["segments"][0]["flags"] == []
    assert report["windows"][0]["scores"] == {"left": 0.65, "middle": 0.8, "right": 0.65}
    assert report["windows"][0]["single_source"] is False


@pytest.mark.parametrize("second_text,score", [("одинаково", 1.0), ("иначе", 0.0)])
def test_ties_keep_engine_input_order(tmp_path, monkeypatch, second_text, score):
    doc, report = run_ensemble(tmp_path, monkeypatch, {
        "zulu": [segment(0, 20, "одинаково")],
        "alpha": [segment(0, 20, second_text)],
    })
    assert doc["segments"][0]["source"] == "zulu"
    assert report["windows"][0]["scores"] == {"zulu": score, "alpha": score}


def test_all_empty_hypotheses_produce_no_segments(tmp_path, monkeypatch):
    doc, report = run_ensemble(tmp_path, monkeypatch, {
        "first": [segment(0, 20, " \n")], "second": [],
    })
    assert doc["segments"] == report["windows"] == []
    assert report["votes"] == {}
    assert report["segments"] == 0
    assert report["mean_confidence"] is None


def test_token_dice_edge_cases():
    assert E.toks("ЁЖ, ёж! A1 а-я") == {"ёж", "a1", "а", "я"}
    assert E.dice(set(), set()) == 1.0
    assert E.dice({"слово"}, set()) == 0.0
    assert E.dice(set(), {"слово"}) == 0.0
    assert E.dice({"один", "два"}, {"два", "три"}) == 0.5


@pytest.mark.parametrize("sr", [8000, 44100, 48000])
def test_x20_rejects_non_16khz_before_loading_models(tmp_path, monkeypatch, sr):
    import numpy as np
    import soundfile as sf

    audio, out = tmp_path / "input.wav", tmp_path / "segments.json"
    sf.write(audio, np.zeros(sr, dtype="float32"), sr)
    monkeypatch.setattr(sys, "argv", ["x20_engine.py", "--engine", "whisper",
                                     "--audio", str(audio), "--out", str(out)])

    def forbidden_load(engine):
        pytest.fail("a model must not be loaded for an unsupported sample rate")

    monkeypatch.setattr(X, "load", forbidden_load)
    with pytest.raises(SystemExit, match="16000"):
        X.main()
    assert not out.exists()


@pytest.fixture
def fake_mlx(monkeypatch):
    """Replace only the ML boundary, while exercising the real adapter closures."""
    import dataclasses
    import types

    @dataclasses.dataclass
    class Result:
        text: str

    events, calls = [], []
    state = {"fail": False}

    def generate(x, **kwargs):
        events.append("infer")
        calls.append(kwargs)
        if state["fail"]:
            raise RuntimeError("synthetic inference failure")
        return Result("распознано")

    def transcribe(x, **kwargs):
        result = generate(x, **kwargs)
        return {"text": result.text}

    names = ["mlx", "mlx.core", "mlx_whisper", "mlx_audio", "mlx_audio.stt",
             "mlx_audio.stt.models", "mlx_audio.stt.models.parakeet",
             "mlx_audio.stt.models.parakeet.parakeet"]
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    core = modules["mlx.core"]
    core.array = X.np.asarray
    core.float32 = X.np.float32
    core.clear_cache = lambda: events.append("clear")
    modules["mlx_whisper"].transcribe = transcribe
    model = types.SimpleNamespace(generate=generate)
    modules["mlx_audio.stt"].load = lambda path: model
    modules["mlx_audio.stt.models.parakeet.parakeet"].Model = types.SimpleNamespace(
        from_pretrained=lambda path, dtype: model)
    return types.SimpleNamespace(events=events, calls=calls, state=state)


@pytest.mark.parametrize("engine", ["whisper", "parakeet", "qwen"])
def test_mlx_cache_is_cleared_after_each_window(engine, fake_mlx):
    run = X.load(engine)
    for _ in range(2):
        assert run(X.np.zeros(16000, dtype="float32"), 16000) == "распознано"
    assert fake_mlx.events == ["infer", "clear", "infer", "clear"]


@pytest.mark.parametrize("engine", ["whisper", "parakeet", "qwen"])
def test_mlx_cache_is_cleared_when_inference_fails(engine, fake_mlx):
    fake_mlx.state["fail"] = True
    run = X.load(engine)
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        run(X.np.zeros(16000, dtype="float32"), 16000)
    assert fake_mlx.events == ["infer", "clear"]


@pytest.mark.parametrize("engine,param,expected", [
    ("whisper", "word_timestamps", False),
    ("qwen", "repetition_penalty", 1.1),
])
def test_x20_generation_parameters_are_explicit(engine, param, expected, fake_mlx):
    X.load(engine)(X.np.zeros(16000, dtype="float32"), 16000)
    assert fake_mlx.calls[0][param] == expected


def test_x20_root_environment_overrides_all_model_paths(tmp_path, monkeypatch, fake_mlx):
    import runpy

    # импорт модуля выше уже выставил ASR_X20_ROOT в дефолт — снимаем его,
    # чтобы проверить именно legacy-алиас X20_ROOT
    monkeypatch.delenv("ASR_X20_ROOT", raising=False)
    monkeypatch.setenv("X20_ROOT", str(tmp_path / "custom-x20"))
    namespace = runpy.run_path(str(ROOT / "engines" / "x20_engine.py"))
    expected = tmp_path / "custom-x20"
    assert namespace["X20"] == expected
    assert namespace["MODELS"] == expected / "work" / "models"
    namespace["load"]("whisper")(X.np.zeros(16000, dtype="float32"), 16000)
    assert fake_mlx.calls[0]["path_or_hf_repo"] == str(expected / "work" / "models" / "whisper")


def test_x20_accepts_16khz_and_clears_each_real_window(tmp_path, monkeypatch, fake_mlx):
    audio, out = tmp_path / "input.wav", tmp_path / "segments.json"
    X.sf.write(audio, X.np.full((40000, 2), 0.25, dtype="float32"), 16000, subtype="FLOAT")
    monkeypatch.setattr(sys, "argv", ["x20_engine.py", "--engine", "whisper",
                                     "--audio", str(audio), "--out", str(out), "--window", "1"])
    assert X.main() == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["audio"]["sr"] == 16000
    assert doc["audio"]["channels"] == 1
    assert [(s["start"], s["end"]) for s in doc["segments"]] == [(0, 1), (1, 2), (2, 2.5)]
    assert fake_mlx.events == ["infer", "clear"] * 3
