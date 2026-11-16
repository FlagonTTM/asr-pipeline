"""Tests for asr_router.plan_route / CLI — no network, no ML, no audio."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asr_router as router  # noqa: E402

DEMO_TRUST = ROOT / "tests" / "fixtures" / "trust_demo.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def reason(code: str, severity: str | None = None, **extra) -> dict:
    row = {"code": code, "detail": f"{code} detail"}
    if severity is not None:
        row["severity"] = severity
    row.update(extra)
    return row


def make_trust(
    reasons: list | None = None,
    *,
    verdict: str | None = "partial",
    duration: float = 100.0,
    peak: float = -20.0,
    clip: float = 0.0,
    union: float = 0.9,
    pair: tuple[str, str] = ("whisper", "kts"),
    windows: list | None = None,
    mean: float | None = None,
    uncovered: list | None = None,
    gaps: list | None = None,
    last_segment_end: float | None = None,
    primary: str = "whisper",
    attempts: dict | None = None,
) -> dict:
    trust = {
        "schema": "asr.trust/2",
        "primary": primary,
        "reasons": [] if reasons is None else reasons,
        "audio_metrics": {"duration_s": duration, "rms_dbfs": -41.0, "peak_dbfs": peak, "clip_fraction": clip},
        "coverage": {
            primary: {
                "coverage": union,
                "coverage_union": union,
                "speech_gaps": gaps or [],
                "last_segment_end": last_segment_end if last_segment_end is not None else duration,
            }
        },
        "coverage_union": {"covered_s": union * duration, "active_s": duration, "coverage_union": union},
        "agreement": {"mean_dice": mean, "windows": windows or [], "pair": list(pair)},
    }
    if verdict is not None:
        trust["verdict"] = verdict
    if uncovered is not None:
        trust["uncovered_all"] = uncovered
    if attempts is not None:
        trust["attempts"] = attempts
    return trust


def step_ids(route: dict) -> list[str]:
    return [step["id"] for step in route["steps"]]


def step(route: dict, step_id: str) -> dict:
    for row in route["steps"]:
        if row["id"] == step_id:
            return row
    raise AssertionError(f"step {step_id!r} not in {step_ids(route)}")


# --------------------------------------------------------------------------- #
# базовый контракт
# --------------------------------------------------------------------------- #
def test_ok_without_reasons_gets_full_summary():
    route = router.plan_route(make_trust([], verdict="ok"))
    assert route["schema"] == "asr.route/1"
    assert route["verdict"] == "ok"
    assert route["codes"] == []
    assert route["blocked"] is None
    assert route["steps"] == []
    assert route["summary_allowed"] is True
    assert route["summary_mode"] == "full"
    assert route["unresolved"] == []
    assert route["budget"] == {"attempts_max": 2, "seconds_max": 35.0}
    assert router.route_exit_code(route) == 0


def test_empty_trust_is_ok_not_crash():
    route = router.plan_route({})
    assert route["verdict"] == "ok"
    assert route["codes"] == []
    assert route["steps"] == []
    assert route["budget"]["seconds_max"] == 0.0


def test_route_shape_is_stable_and_serialisable():
    route = router.plan_route(make_trust([reason("VAD_UNRELIABLE")]))
    assert set(route) == {
        "schema",
        "verdict",
        "codes",
        "blocked",
        "steps",
        "budget",
        "summary_allowed",
        "summary_mode",
        "unresolved",
        "notes",
    }
    assert set(route["budget"]) == {"attempts_max", "seconds_max"}
    for row in route["steps"]:
        assert set(row) == {"id", "kind", "because", "params"}
    json.dumps(route, ensure_ascii=False)


def test_plan_route_does_not_mutate_inputs():
    trust = make_trust([reason("SPEECH_GAP", intervals=[{"start": 1.0, "end": 5.0}]), reason("LOW_LEVEL", peak=-40)])
    cfg = {"budget_frac": 0.2}
    trust_before, cfg_before = copy.deepcopy(trust), copy.deepcopy(cfg)
    router.plan_route(trust, cfg)
    assert trust == trust_before
    assert cfg == cfg_before


# --------------------------------------------------------------------------- #
# LOW_LEVEL
# --------------------------------------------------------------------------- #
def test_low_level_plans_normalize_gain():
    route = router.plan_route(make_trust([reason("LOW_LEVEL")], peak=-20.0))
    row = step(route, "normalize")
    assert row["kind"] == "gain"
    assert row["because"] == "LOW_LEVEL"
    assert row["params"]["gain_db"] == 19.0  # -1 - (-20)
    assert row["params"]["target_peak_dbfs"] == -1.0
    assert route["verdict"] == "partial"


def test_low_level_gain_is_capped_at_40_db():
    route = router.plan_route(make_trust([reason("LOW_LEVEL")], peak=-60.0))
    assert step(route, "normalize")["params"]["gain_db"] == 40.0
    assert any("40" in note for note in route["notes"])


def test_low_level_skipped_when_engine_not_level_sensitive():
    trust = make_trust([reason("LOW_LEVEL")], peak=-20.0)
    route = router.plan_route(trust, {"level_sensitive": False})
    assert "normalize" not in step_ids(route)
    assert "LOW_LEVEL" in route["unresolved"]
    assert any("level_sensitive" in note for note in route["notes"])


def test_low_level_skipped_from_engines_map():
    trust = make_trust([reason("LOW_LEVEL")], peak=-20.0)
    route = router.plan_route(trust, {"engines": {"whisper": {"level_sensitive": False}}})
    assert route["steps"] == []
    assert "LOW_LEVEL" in route["unresolved"]


def test_low_level_without_peak_is_unresolved():
    trust = make_trust([reason("LOW_LEVEL")])
    del trust["audio_metrics"]["peak_dbfs"]
    route = router.plan_route(trust)
    assert "normalize" not in step_ids(route)
    assert "LOW_LEVEL" in route["unresolved"]


# --------------------------------------------------------------------------- #
# CLIPPING
# --------------------------------------------------------------------------- #
def test_clipping_below_threshold_plans_no_gain():
    route = router.plan_route(make_trust([reason("CLIPPING")], clip=0.002))
    row = step(route, "no_gain")
    assert row["kind"] == "gain_guard"
    assert row["params"]["gain_db"] == 0.0
    assert row["params"]["clip_fraction"] == 0.002
    assert route["blocked"] is None
    assert route["verdict"] == "partial"
    assert route["summary_mode"] == "caveated"


def test_clipping_above_threshold_blocks():
    route = router.plan_route(make_trust([reason("CLIPPING")], clip=0.01))
    assert route["verdict"] == "failed"
    assert route["blocked"]["code"] == "CLIPPING"
    assert route["blocked"]["codes"] == ["CLIPPING"]
    assert route["steps"] == []
    assert route["summary_allowed"] is False
    assert route["summary_mode"] == "blocked"
    assert router.route_exit_code(route) == 4


def test_clipping_threshold_boundary_is_not_blocked():
    route = router.plan_route(make_trust([reason("CLIPPING")], clip=0.005))
    assert route["blocked"] is None
    assert step(route, "no_gain")["params"]["gain_db"] == 0.0


# --------------------------------------------------------------------------- #
# VAD_UNRELIABLE
# --------------------------------------------------------------------------- #
def test_vad_unreliable_switches_detector_to_energy():
    route = router.plan_route(make_trust([reason("VAD_UNRELIABLE")]))
    row = step(route, "detector")
    assert row["kind"] == "vad"
    assert row["because"] == "VAD_UNRELIABLE"
    assert row["params"] == {"detector": "energy"}


# --------------------------------------------------------------------------- #
# LOW_COVERAGE
# --------------------------------------------------------------------------- #
def test_low_coverage_requests_second_engine_with_peer_name():
    route = router.plan_route(make_trust([reason("LOW_COVERAGE")], union=0.5))
    row = step(route, "second_engine")
    assert row["kind"] == "engine"
    assert row["because"] == "LOW_COVERAGE"
    assert row["params"] == {"engine": "kts", "min_union": 0.65}
    assert route["blocked"] is None


def test_low_coverage_blocks_after_repeat_below_union():
    route = router.plan_route(make_trust([reason("LOW_COVERAGE")], union=0.5), {"second_engine_done": True})
    assert route["verdict"] == "failed"
    assert route["blocked"]["code"] == "LOW_COVERAGE"
    assert route["steps"] == []
    assert router.route_exit_code(route) == 4


def test_low_coverage_after_repeat_above_union_needs_no_step():
    route = router.plan_route(make_trust([reason("LOW_COVERAGE")], union=0.7), {"second_engine_done": True})
    assert "second_engine" not in step_ids(route)
    assert "LOW_COVERAGE" in route["unresolved"]
    assert route["blocked"] is None


def test_low_coverage_retry_flag_from_trust_attempts():
    route = router.plan_route(make_trust([reason("LOW_COVERAGE")], union=0.4, attempts={"second_engine": 1}))
    assert route["blocked"]["code"] == "LOW_COVERAGE"


def test_fill_from_peers_is_opt_in_and_ordered():
    trust = make_trust([reason("LOW_COVERAGE")], union=0.5)
    route = router.plan_route(trust, {"fill_from_peers": True})
    assert step_ids(route) == ["second_engine", "fill_from_peers"]
    assert step(route, "fill_from_peers")["params"]["engines"] == ["kts"]


# --------------------------------------------------------------------------- #
# TAIL_UNCOVERED / SPEECH_GAP -> repair_decode
# --------------------------------------------------------------------------- #
def test_tail_uncovered_uses_reason_intervals():
    intervals = [{"start": 90.0, "end": 95.0}]
    route = router.plan_route(make_trust([reason("TAIL_UNCOVERED", intervals=intervals)]))
    row = step(route, "repair_decode")
    assert row["kind"] == "repair"
    assert row["because"] == "TAIL_UNCOVERED"
    assert row["params"]["intervals"] == [{"start": 90.0, "end": 95.0}]


def test_speech_gap_uses_uncovered_all_when_reason_has_no_intervals():
    trust = make_trust(
        [reason("SPEECH_GAP")],
        uncovered=[{"start": 40.2, "end": 54.9}, {"start": 70.0, "end": 71.0}],
    )
    route = router.plan_route(trust)
    assert step(route, "repair_decode")["params"]["intervals"] == [
        {"start": 40.2, "end": 54.9},
        {"start": 70.0, "end": 71.0},
    ]


def test_repair_merges_overlaps_and_drops_short_holes():
    intervals = [
        {"start": 1.0, "end": 2.0},
        {"start": 1.5, "end": 3.0},
        {"start": 5.0, "end": 5.5},  # < 1.0 s — не ремонтопригодна
        {"start": 10.0, "end": 12.0},
    ]
    route = router.plan_route(make_trust([reason("SPEECH_GAP", intervals=intervals)]))
    assert step(route, "repair_decode")["params"]["intervals"] == [
        {"start": 1.0, "end": 3.0},
        {"start": 10.0, "end": 12.0},
    ]


def test_repair_falls_back_to_coverage_speech_gaps():
    trust = make_trust([reason("SPEECH_GAP")], gaps=[{"start": 40.53, "end": 54.62}])
    route = router.plan_route(trust)
    assert step(route, "repair_decode")["params"]["intervals"] == [{"start": 40.53, "end": 54.62}]


def test_tail_uncovered_adds_interval_after_last_segment():
    trust = make_trust([reason("TAIL_UNCOVERED")], duration=100.0, last_segment_end=90.0, gaps=[])
    route = router.plan_route(trust)
    assert step(route, "repair_decode")["params"]["intervals"] == [{"start": 90.0, "end": 100.0}]


def test_repair_without_any_interval_is_unresolved():
    trust = make_trust([reason("SPEECH_GAP")], last_segment_end=100.0, gaps=[])
    route = router.plan_route(trust)
    assert "repair_decode" not in step_ids(route)
    assert "SPEECH_GAP" in route["unresolved"]


# --------------------------------------------------------------------------- #
# LOW_AGREEMENT -> arbitrate
# --------------------------------------------------------------------------- #
def test_low_agreement_arbitrates_only_bad_windows():
    windows = [
        {"start": 0, "end": 20, "dice": 0.6},
        {"start": 20, "end": 40, "dice": 0.2},
        {"start": 40, "end": 60, "dice": 0.35},
    ]
    route = router.plan_route(make_trust([reason("LOW_AGREEMENT")], windows=windows, mean=0.28))
    row = step(route, "arbitrate")
    assert row["kind"] == "arbitration"
    assert row["params"]["max_dice"] == 0.35
    assert row["params"]["windows"] == [{"start": 20.0, "end": 40.0, "dice": 0.2}]


def test_low_agreement_without_windows_uses_whole_file():
    route = router.plan_route(make_trust([reason("LOW_AGREEMENT")], windows=[], mean=0.2))
    assert step(route, "arbitrate")["params"]["windows"] == [{"start": 0.0, "end": 100.0, "dice": 0.2}]


def test_low_agreement_without_data_is_unresolved():
    route = router.plan_route(make_trust([reason("LOW_AGREEMENT")], windows=[], mean=None))
    assert "arbitrate" not in step_ids(route)
    assert "LOW_AGREEMENT" in route["unresolved"]


# --------------------------------------------------------------------------- #
# блокировки
# --------------------------------------------------------------------------- #
def test_language_mismatch_blocks_and_forbids_summary():
    route = router.plan_route(make_trust([reason("LANGUAGE_MISMATCH")]))
    assert route["verdict"] == "failed"
    assert route["blocked"]["code"] == "LANGUAGE_MISMATCH"
    assert route["blocked"]["codes"] == ["LANGUAGE_MISMATCH"]
    assert route["steps"] == []
    assert route["summary_allowed"] is False
    assert route["summary_mode"] == "blocked"
    assert route["unresolved"] == []
    assert router.route_exit_code(route) == 4


def test_language_mismatch_blocks_even_with_repairable_codes():
    trust = make_trust([reason("LANGUAGE_MISMATCH"), reason("SPEECH_GAP", intervals=[{"start": 1, "end": 5}])])
    route = router.plan_route(trust)
    assert route["blocked"]["code"] == "LANGUAGE_MISMATCH"
    assert route["blocked"]["codes"] == ["LANGUAGE_MISMATCH"]
    assert route["steps"] == []
    assert route["summary_allowed"] is False


def test_unknown_code_is_fail_closed():
    route = router.plan_route(make_trust([reason("MYSTERY_CODE")]))
    assert route["verdict"] == "failed"
    assert route["blocked"]["code"] == "UNKNOWN_REASON"
    assert route["blocked"]["codes"] == ["MYSTERY_CODE"]
    assert route["steps"] == []
    assert route["summary_allowed"] is False
    assert "UNKNOWN_REASON" in json.dumps(route["blocked"], ensure_ascii=False)


def test_unknown_code_wins_over_known_codes_and_lists_them():
    route = router.plan_route(make_trust([reason("LOW_LEVEL"), reason("ALIEN_REASON")]))
    assert route["blocked"]["code"] == "UNKNOWN_REASON"
    assert route["blocked"]["codes"] == ["ALIEN_REASON"]
    assert route["steps"] == []
    assert route["codes"] == ["LOW_LEVEL", "ALIEN_REASON"]


def test_unknown_severity_is_treated_as_block():
    route = router.plan_route(make_trust([reason("VAD_UNRELIABLE", severity="fatal")]))
    assert route["verdict"] == "failed"
    assert route["summary_mode"] == "blocked"


def test_block_severity_raises_failed_but_keeps_summary_flag_rules():
    trust = make_trust([reason("SPEECH_GAP", severity="block", intervals=[{"start": 1.0, "end": 5.0}])])
    route = router.plan_route(trust)
    assert route["verdict"] == "failed"
    assert route["summary_mode"] == "blocked"
    assert route["summary_allowed"] is False
    assert "repair_decode" in step_ids(route)  # шаг остаётся видимым, но вердикт failed


# --------------------------------------------------------------------------- #
# несколько кодов и порядок шагов
# --------------------------------------------------------------------------- #
def test_multiple_codes_keep_fixed_step_order():
    trust = make_trust(
        [
            reason("LOW_AGREEMENT"),
            reason("LOW_LEVEL"),
            reason("CLIPPING"),
            reason("VAD_UNRELIABLE"),
            reason("LOW_COVERAGE"),
            reason("SPEECH_GAP", intervals=[{"start": 40.0, "end": 55.0}]),
        ],
        peak=-30.0,
        clip=0.0,
        union=0.5,
        windows=[{"start": 0, "end": 20, "dice": 0.2}],
        mean=0.28,
    )
    route = router.plan_route(trust)
    assert step_ids(route) == [
        "normalize",
        "detector",
        "no_gain",
        "second_engine",
        "repair_decode",
        "arbitrate",
    ]
    positions = [router.STEP_ORDER.index(step_id) for step_id in step_ids(route)]
    assert positions == sorted(positions)
    assert route["codes"] == [
        "LOW_AGREEMENT",
        "LOW_LEVEL",
        "CLIPPING",
        "VAD_UNRELIABLE",
        "LOW_COVERAGE",
        "SPEECH_GAP",
    ]


def test_steps_contain_only_required_actions():
    route = router.plan_route(make_trust([reason("VAD_UNRELIABLE")]))
    assert step_ids(route) == ["detector"]
    route = router.plan_route(make_trust([reason("LOW_LEVEL")], peak=-20.0))
    assert step_ids(route) == ["normalize"]
    route = router.plan_route(make_trust([reason("SPEECH_GAP", intervals=[{"start": 1, "end": 5}])]))
    assert step_ids(route) == ["repair_decode"]


def test_summarize_step_is_opt_in_and_last():
    route = router.plan_route(make_trust([reason("LOW_LEVEL")], peak=-20.0), {"summarize": True})
    assert step_ids(route) == ["normalize", "summarize"]
    assert step(route, "summarize")["params"]["mode"] == "caveated"


# --------------------------------------------------------------------------- #
# бюджет
# --------------------------------------------------------------------------- #
def test_budget_rounds_to_tenth():
    route = router.plan_route(make_trust([], verdict="ok", duration=178.25))
    assert route["budget"] == {"attempts_max": 2, "seconds_max": 62.4}


def test_budget_honours_budget_frac():
    route = router.plan_route(make_trust([], verdict="ok", duration=100.0), {"budget_frac": 0.1})
    assert route["budget"]["seconds_max"] == 10.0
    assert route["budget"]["attempts_max"] == 2


def test_budget_rejects_bad_frac():
    with pytest.raises(router.RouteError) as exc:
        router.plan_route(make_trust([], verdict="ok"), {"budget_frac": 0})
    assert "ROUTE_BAD_CFG" in str(exc.value)


# --------------------------------------------------------------------------- #
# реальный trust из гейта
# --------------------------------------------------------------------------- #
def test_demo_fixture_from_gate_is_routed():
    assert DEMO_TRUST.is_file(), f"missing fixture: {DEMO_TRUST}"
    trust = json.loads(DEMO_TRUST.read_text(encoding="utf-8"))
    route = router.plan_route(trust)
    assert route["schema"] == "asr.route/1"
    assert route["verdict"] == "partial"
    assert route["codes"] == ["VAD_UNRELIABLE", "SPEECH_GAP", "LOW_AGREEMENT"]
    assert route["budget"] == {"attempts_max": 2, "seconds_max": 62.4}
    assert step_ids(route) == ["detector", "repair_decode", "arbitrate"]
    assert step(route, "detector")["params"]["detector"] == "energy"
    intervals = step(route, "repair_decode")["params"]["intervals"]
    assert len(intervals) == 6
    assert intervals[0] == {"start": 0.0, "end": 15.03}
    windows = step(route, "arbitrate")["params"]["windows"]
    assert all(window["dice"] < 0.35 for window in windows)
    assert route["summary_allowed"] is True
    assert route["summary_mode"] == "caveated"
    assert route["unresolved"] == []
    assert router.route_exit_code(route) == 2


def test_demo_fixture_with_budget_override():
    trust = json.loads(DEMO_TRUST.read_text(encoding="utf-8"))
    route = router.plan_route(trust, {"budget_frac": 0.1})
    assert route["budget"]["seconds_max"] == 17.8


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_writes_route_file_and_exits_partial(tmp_path, capsys):
    out = tmp_path / "route.json"
    code = router.main(["--trust", str(DEMO_TRUST), "--out", str(out)])
    assert code == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == "asr.route/1"
    assert printed == json.loads(out.read_text(encoding="utf-8"))


def test_cli_treats_out_directory_as_dir(tmp_path):
    out_dir = tmp_path / "plans"
    code = router.main(["--trust", str(DEMO_TRUST), "--out", str(out_dir)])
    assert code == 2
    assert (out_dir / "route.json").is_file()


def test_cli_blocked_exit_code(tmp_path, capsys):
    trust = make_trust([reason("LANGUAGE_MISMATCH")])
    src = tmp_path / "trust.json"
    src.write_text(json.dumps(trust, ensure_ascii=False), encoding="utf-8")
    code = router.main(["--trust", str(src), "--out", str(tmp_path / "route.json")])
    assert code == 4
    route = json.loads((tmp_path / "route.json").read_text(encoding="utf-8"))
    assert route["blocked"]["code"] == "LANGUAGE_MISMATCH"


def test_cli_bad_args_return_64(capsys):
    code = router.main([])
    assert code == 64
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "ROUTE_BAD_ARGS"


def test_cli_missing_trust_file_returns_64(tmp_path, capsys):
    code = router.main(["--trust", str(tmp_path / "nope.json"), "--out", str(tmp_path / "route.json")])
    assert code == 64
    assert json.loads(capsys.readouterr().out)["error"] == "ROUTE_BAD_ARGS"


def test_cli_bad_json_returns_64(tmp_path, capsys):
    src = tmp_path / "trust.json"
    src.write_text("{not json", encoding="utf-8")
    code = router.main(["--trust", str(src), "--out", str(tmp_path / "route.json")])
    assert code == 64
    assert json.loads(capsys.readouterr().out)["error"] == "ROUTE_BAD_TRUST"


def test_cli_budget_frac_flag(tmp_path, capsys):
    src = tmp_path / "trust.json"
    src.write_text(json.dumps(make_trust([], verdict="ok", duration=200.0)), encoding="utf-8")
    code = router.main(["--trust", str(src), "--out", str(tmp_path / "r.json"), "--budget-frac", "0.25"])
    assert code == 0
    route = json.loads(capsys.readouterr().out)
    assert route["budget"]["seconds_max"] == 50.0


# --------------------------------------------------------------------------- #
# устойчивость входа
# --------------------------------------------------------------------------- #
def test_reason_without_code_is_rejected():
    with pytest.raises(router.RouteError) as exc:
        router.plan_route({"reasons": [{"detail": "no code"}]})
    assert "ROUTE_BAD_TRUST" in str(exc.value)


def test_trust_must_be_object():
    with pytest.raises(router.RouteError):
        router.plan_route(["not", "a", "trust"])


def test_string_reason_is_accepted():
    route = router.plan_route({"reasons": ["LOW_LEVEL"], "audio_metrics": {"duration_s": 10, "peak_dbfs": -21}})
    assert route["codes"] == ["LOW_LEVEL"]
    assert step_ids(route) == ["normalize"]
