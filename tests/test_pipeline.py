"""Tests for asr_pipeline.py (orchestrator) and adapt_legacy.py.

No ML and no network: the ASR engine and the three stage modules that are still
being written by other agents are replaced by stub executables placed in a
temporary `--stages-dir`; the orchestrator is exercised through its real CLI.
A separate contract probe skips (with a message) while a real stage module is
missing or does not accept the contract flags yet.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import adapt_legacy as AL  # noqa: E402

PY = sys.executable
FIXTURES = ROOT / "fixtures"
WHISPER = json.loads((FIXTURES / "whisper.json").read_text(encoding="utf-8"))["segments"]
KTS = FIXTURES / "kts.json"
DURATION = 180.0
STAGE_NAMES = ["ingest", "asr", "gate", "route", "repair_plan", "repair_run",
               "repair_merge", "gate_after", "summary", "verify"]

# --------------------------------------------------------------------- stubs

ENGINE_STUB = r'''
import argparse, json, os, shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--out", required=True)
p.add_argument("--language", default="ru")
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
log = os.environ.get("ASR_ORDER_LOG")
if log:
    with open(log, "a", encoding="utf-8") as f:
        f.write((os.environ.get("ASR_STAGE") or "asr") + "\n")
shutil.copyfile(Path(os.environ["ASR_TEST_WHISPER"]), out / "segments.json")
(out / "language.txt").write_text(a.language, encoding="utf-8")
print(json.dumps({"engine": "stub", "out": str(out)}))
'''

GATE_STUB = r'''
import argparse, json, os, sys, wave
from pathlib import Path

def union(rows):
    out = []
    for row in sorted(rows, key=lambda r: r["start"]):
        if out and row["start"] <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], row["end"])
        else:
            out.append([row["start"], row["end"]])
    return out

def duration(path):
    with wave.open(path, "rb") as handle:
        return handle.getnframes() / handle.getframerate()

p = argparse.ArgumentParser()
p.add_argument("--audio", required=True)
p.add_argument("--transcript", action="append", default=[])
p.add_argument("--evidence", action="append", default=[])
p.add_argument("--primary", default="whisper")
p.add_argument("--vad", default="silero")
p.add_argument("--lid", default="whisper")
p.add_argument("--language", default="ru")
p.add_argument("--out", required=True)
a = p.parse_args()
log = os.environ.get("ASR_ORDER_LOG")
if log:
    with open(log, "a", encoding="utf-8") as f:
        f.write((os.environ.get("ASR_STAGE") or "gate") + "\n")

total = duration(a.audio)
segments = []
for spec in a.transcript:
    name, _, path = spec.partition("=")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("segments") if isinstance(data, dict) else data
    if name == a.primary:
        segments = [r for r in rows or [] if r.get("text")]

spans = union([{"start": float(r["start"]), "end": float(r["end"])} for r in segments])
covered = sum(b - a0 for a0, b in spans)
gaps, cursor = [], 0.0
for a0, b0 in spans:
    if a0 - cursor >= 1.0:
        gaps.append({"start": round(cursor, 2), "end": round(a0, 2)})
    cursor = max(cursor, b0)
if total - cursor >= 1.0:
    gaps.append({"start": round(cursor, 2), "end": round(total, 2)})

verdict = os.environ.get("ASR_STUB_VERDICT", "partial")
severity = "block" if verdict == "failed" else "warn"
reasons = []
if verdict != "ok":
    if gaps:
        reasons.append({"code": "SPEECH_GAP", "severity": severity, "detail": "stub gaps",
                        "intervals": gaps})
    else:
        reasons.append({"code": "LOW_COVERAGE", "severity": severity, "detail": "stub", "intervals": []})

tail = round(total - spans[-1][1], 1) if spans else round(total, 1)
report = {
    "schema": "asr.trust/2", "verdict": verdict, "primary": a.primary, "gate_version": "stub",
    "reasons": reasons,
    "audio_metrics": {"duration_s": round(total, 2), "rms_dbfs": -41.58, "peak_dbfs": -13.09,
                      "clip_fraction": 0.0, "band_share_below_1khz": 0.75, "rms_per_10s": []},
    "activity": {"energy_active_s": round(total, 1), "vad_active_s": round(total, 1),
                 "noise_floor_dbfs": -83.8},
    "language_id": {"language": a.language, "probability": 0.97, "method": "declared"},
    "coverage": {a.primary: {"coverage": round(covered / total, 3),
                             "coverage_union": round(covered / total, 3),
                             "covered_s": round(covered, 1), "union_active_s": round(total, 1),
                             "tail_uncovered_s": tail, "speech_gaps": gaps}},
    "coverage_union": {"covered_s": round(covered, 1), "active_s": round(total, 1),
                       "coverage_union": round(covered / total, 3)},
    "uncovered_all": gaps,
    "agreement": {"mean_dice": 0.28, "windows": [], "pair": []},
    "config": {"vad": a.vad, "lid": a.lid, "thresholds": {}},
}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "trust.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"verdict": verdict, "reasons": [r["code"] for r in reasons]}))
sys.exit(0 if verdict == "ok" else 2)
'''

ROUTER_STUB = r'''
import argparse, json, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--trust", required=True)
p.add_argument("--out", required=True)
p.add_argument("--budget-frac", type=float, default=0.35)
a = p.parse_args()
log = os.environ.get("ASR_ORDER_LOG")
if log:
    with open(log, "a", encoding="utf-8") as f:
        f.write((os.environ.get("ASR_STAGE") or "route") + "\n")

trust = json.loads(Path(a.trust).read_text(encoding="utf-8"))
verdict = trust.get("verdict", "partial")
codes = [r.get("code") for r in trust.get("reasons") or []]
total = float((trust.get("audio_metrics") or {}).get("duration_s") or 0.0)
blocked = None
summary_allowed = verdict != "failed"
if os.environ.get("ASR_STUB_ROUTE") == "blocked" or "LANGUAGE_MISMATCH" in codes:
    blocked = {"code": "LANGUAGE_MISMATCH", "detail": "stub"}
    summary_allowed = False
steps = []
if trust.get("uncovered_all") and not blocked:
    steps.append({"id": "repair_decode", "kind": "repair", "because": "SPEECH_GAP",
                  "params": {"intervals": trust["uncovered_all"]}})
route = {"schema": "asr.route/1", "verdict": "blocked" if blocked else verdict,
         "codes": codes, "blocked": blocked, "steps": steps,
         "budget": {"attempts_max": int(os.environ.get("ASR_STUB_ATTEMPTS", "1")),
                    "seconds_max": round(a.budget_frac * total)},
         "summary_allowed": summary_allowed,
         "summary_mode": "caveated" if verdict == "partial" else "plain",
         "unresolved": [], "notes": []}
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(route, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"verdict": route["verdict"], "blocked": bool(blocked)}))
'''

REPAIR_STUB = r'''
import argparse, json, os
from pathlib import Path

log = os.environ.get("ASR_ORDER_LOG")
if log:
    with open(log, "a", encoding="utf-8") as f:
        f.write((os.environ.get("ASR_STAGE") or "repair") + "\n")

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="mode", required=True)
pp = sub.add_parser("plan")
pp.add_argument("--trust", required=True)
pp.add_argument("--route", required=True)
pp.add_argument("--out", required=True)
pr = sub.add_parser("run")
pr.add_argument("--audio", required=True)
pr.add_argument("--plan", required=True)
pr.add_argument("--engines", required=True)
pr.add_argument("--out", required=True)
pm = sub.add_parser("merge")
pm.add_argument("--base", required=True)
pm.add_argument("--additions", required=True)
pm.add_argument("--out", required=True)
a = p.parse_args()

if a.mode == "plan":
    trust = json.loads(Path(a.trust).read_text(encoding="utf-8"))
    route = json.loads(Path(a.route).read_text(encoding="utf-8"))
    total = float((trust.get("audio_metrics") or {}).get("duration_s") or 0.0)
    budget = float((route.get("budget") or {}).get("seconds_max") or round(0.35 * total))
    gaps = [i for i in (trust.get("uncovered_all") or []) if i["end"] - i["start"] >= 1.0]
    intervals = [{"id": "r%d" % (n + 1), "start": round(i["start"], 2), "end": round(i["end"], 2),
                  "pad_s": 0.3, "reason": "gap", "engines": ["whisper"],
                  "decode_s": round(i["end"] - i["start"], 1)} for n, i in enumerate(gaps)]
    decode = round(sum(i["decode_s"] for i in intervals), 1)
    plan = {"schema": "asr.repair.plan/1", "attempt": 1, "intervals": intervals,
            "total_decode_s": decode, "budget_s": budget, "within_budget": decode <= budget,
            "stop_rule": {"min_gain_ratio": 0.25, "max_attempts": 1}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"intervals": len(intervals)}))
elif a.mode == "run":
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    partial = os.environ.get("ASR_STUB_PARTIAL") == "1"
    rows = []
    for i in plan.get("intervals") or []:
        start, end = round(i["start"], 2), round(i["end"], 2)
        if partial:  # cover only 60% of the hole: forces a second attempt
            end = round(start + 0.6 * (end - start), 2)
        rows.append({"start": start, "end": end, "text": "repair stub " + i["id"]})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "additions.json").write_text(
        json.dumps({"schema": "asr.segments/1", "segments": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps({"added": len(rows)}))
else:
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    add = json.loads(Path(a.additions).read_text(encoding="utf-8"))
    rows = list(base.get("segments") or []) + list(add.get("segments") or [])
    rows.sort(key=lambda r: (float(r["start"]), float(r["end"])))
    for n, row in enumerate(rows, 1):
        row["id"] = "s%04d" % n
    merged = dict(base)
    merged["segments"] = rows
    meta = dict(merged.get("meta") or {})
    meta["merged_additions"] = len(add.get("segments") or [])
    merged["meta"] = meta
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"segments": len(rows)}))
'''

SUMMARY_STUB = r'''
import argparse, hashlib, json, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["ASR_TEST_ROOT"])
from asr_summary import quality_lines

p = argparse.ArgumentParser()
p.add_argument("--trust", required=True)
p.add_argument("--transcript", required=True)
p.add_argument("--segments")
p.add_argument("--out", required=True)
a = p.parse_args()
log = os.environ.get("ASR_ORDER_LOG")
if log:
    with open(log, "a", encoding="utf-8") as f:
        f.write((os.environ.get("ASR_STAGE") or "summary") + "\n")

trust = json.loads(Path(a.trust).read_text(encoding="utf-8"))
verdict = trust.get("verdict", "partial")
codes = [r.get("code") for r in trust.get("reasons") or []]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
lines = ["# Саммари встречи", ""]
if verdict == "partial":
    lines += ["⚠ Часть текста сомнительна: " + ", ".join(codes), ""]
lines += ["## Качество записи", "", *quality_lines(trust),
          "", "## О чём встреча", "", "Заглушка.", "", "## Решения", "", "—", "",
          "## Задачи", "", "— [00:00] «Заглушка»", "", "## Спорные места", ""]
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
payload = {"schema": "asr.summary/1", "trusted": verdict == "ok", "verdict": verdict, "codes": codes,
           "guard": {"numbers_ok": True, "tasks_quoted": True, "caveat_present": True},
           "numbers_sha256": hashlib.sha256(b"stub").hexdigest()}
if os.environ.get("ASR_STUB_REFUSE") == "1":
    payload["status"] = "refused"
    payload["guard"]["numbers_ok"] = False
    (out / "summary.md").rename(out / "summary.rejected.md")
(out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(payload))
if payload.get("status") == "refused":
    raise SystemExit(4)
'''

def union_gaps(rows, duration, min_gap=1.0):
    """Activity gaps the stub gate will report for a transcript (same rule as trust)."""
    spans = []
    for row in sorted(rows, key=lambda r: float(r["start"])):
        start, end = float(row["start"]), float(row["end"])
        if spans and start <= spans[-1][1] + 1e-9:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    gaps, cursor = [], 0.0
    for start, end in spans:
        if start - cursor >= min_gap:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= min_gap:
        gaps.append((cursor, duration))
    return gaps


EXPECTED_GAPS = union_gaps(WHISPER, DURATION)

ARTIFACTS = [
    "run.json",
    "00_input/manifest.json",
    "00_input/engines.json",
    "10_asr/engine/segments.json",
    "10_asr/engine/language.txt",
    "10_asr/segments.json",
    "20_gate/trust.json",
    "30_route/route.json",
    "40_repair/plan.json",
    "40_repair/additions.json",
    "40_repair/segments_merged.json",
    "21_gate_after/trust.json",
    "50_summary/summary.md",
    "50_summary/summary.json",
    "60_verify/report.json",
    "logs/gate.log",
]

# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def audio(tmp_path_factory):
    """Hermetic 180 s 16 kHz mono wav (long enough for fixtures/whisper.json)."""
    path = tmp_path_factory.mktemp("audio") / "talk.wav"
    block = b"\xe8\x03" * 16000  # constant level: energy fallback marks the whole file active
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        for _ in range(int(DURATION)):
            handle.writeframes(block)
    return path


@pytest.fixture(scope="session")
def rig(tmp_path_factory):
    base = tmp_path_factory.mktemp("rig")
    stages = base / "stages"
    stages.mkdir()
    for name, source in {"asr_gate.py": GATE_STUB, "asr_router.py": ROUTER_STUB,
                         "asr_repair.py": REPAIR_STUB, "asr_summary.py": SUMMARY_STUB}.items():
        (stages / name).write_text(source, encoding="utf-8")
    shutil.copyfile(ROOT / "adapt_legacy.py", stages / "adapt_legacy.py")
    engine = base / "engine_stub.py"
    engine.write_text(ENGINE_STUB, encoding="utf-8")
    engines = base / "engines.json"
    engines.write_text(json.dumps({"whisper": {
        "cmd": [PY, str(engine), "--audio", "{audio}", "--out", "{out}", "--language", "{language}"],
        "model": "large-v3", "level_sensitive": True, "timeout_s": 120}}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    order_log = base / "order.log"
    order_log.write_text("", encoding="utf-8")
    return SimpleNamespace(base=base, stages=stages, engines=engines, engine=engine,
                           order_log=order_log)


def stage_env(rig, env_extra=None):
    env = dict(os.environ)
    env["ASR_ORDER_LOG"] = str(rig.order_log)
    env["ASR_TEST_WHISPER"] = str(FIXTURES / "whisper.json")
    env["ASR_TEST_ROOT"] = str(ROOT)
    env.update(env_extra or {})
    return env


def pipeline(rig, args, env_extra=None):
    return subprocess.run([PY, str(ROOT / "asr_pipeline.py"), *[str(a) for a in args]],
                          capture_output=True, text=True, env=stage_env(rig, env_extra),
                          cwd=str(ROOT), timeout=300)


def reset_log(rig):
    rig.order_log.write_text("", encoding="utf-8")


def read_log(rig):
    return [line.strip() for line in rig.order_log.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def run_args(rig, audio, out, *extra):
    return ["run", "--audio", str(audio), "--out", str(out), "--engines", str(rig.engines),
            "--stages-dir", str(rig.stages), "--peer", f"kts={KTS}", *extra]


@pytest.fixture(scope="session")
def base_run(rig, audio, tmp_path_factory):
    out = tmp_path_factory.mktemp("base_run")
    reset_log(rig)
    proc = pipeline(rig, run_args(rig, audio, out))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    return SimpleNamespace(dir=out, proc=proc, audio=audio, order=read_log(rig))


def copy_run(base_run, tmp_path, name="run"):
    target = tmp_path / name
    shutil.copytree(base_run.dir, target)
    return target


# ------------------------------------------------------------------- tests

def test_full_run_exit_and_run_json(base_run):
    payload = json.loads(base_run.proc.stdout)
    assert payload["verdict"] == "partial"
    assert payload["exit_code"] == 2
    assert payload["verify_ok"] is True

    run = json.loads((base_run.dir / "run.json").read_text(encoding="utf-8"))
    assert run["schema"] == "asr.run/1"
    assert [stage["name"] for stage in run["stages"]] == STAGE_NAMES
    assert [stage["status"] for stage in run["stages"]] == ["done"] * len(STAGE_NAMES)
    assert run["attempts"]["repair"] == 1
    assert run["engines"]["whisper"]["model"] == "large-v3"
    assert run["verdict"] == "partial" and run["exit_code"] == 2
    assert run["effective_transcript"] == "40_repair/segments_merged.json"
    assert all(stage["inputs_hash"] for stage in run["stages"])


def test_stage_execution_order(base_run):
    # every contract module is a subprocess; stubs log ASR_STAGE as they run
    assert base_run.order == ["asr", "gate", "route", "repair_plan", "repair_run",
                              "repair_merge", "gate_after", "summary"]
    assert "ingest" not in base_run.order and "verify" not in base_run.order


def test_artifacts_exist(base_run):
    missing = [rel for rel in ARTIFACTS if not (base_run.dir / rel).exists()]
    assert not missing, missing


def test_segments_document(base_run):
    manifest = json.loads((base_run.dir / "00_input" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "asr.input/1"
    assert manifest["probe"] in {"ffprobe", "soundfile"}
    assert manifest["audio"]["duration_s"] == pytest.approx(DURATION, abs=0.01)
    assert manifest["audio"]["sr"] == 16000 and manifest["audio"]["channels"] == 1

    doc = json.loads((base_run.dir / "10_asr" / "segments.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "asr.segments/1"
    assert len(doc["segments"]) == len(WHISPER) == 16
    assert doc["segments"][0]["id"] == "s0001"
    assert doc["audio"]["sha256"] == manifest["audio"]["sha256"]
    assert doc["engine"]["name"] == "whisper" and doc["engine"]["model"] == "large-v3"
    assert (base_run.dir / "10_asr" / "engine" / "language.txt").read_text(encoding="utf-8") == "ru"

    merged = json.loads((base_run.dir / "40_repair" / "segments_merged.json").read_text(encoding="utf-8"))
    assert len(merged["segments"]) == len(WHISPER) + len(EXPECTED_GAPS)  # one addition per union hole
    assert len(EXPECTED_GAPS) == 4
    starts = [float(segment["start"]) for segment in merged["segments"]]
    assert starts == sorted(starts)


def test_unresolved_matches_trust(base_run):
    final = json.loads((base_run.dir / "21_gate_after" / "trust.json").read_text(encoding="utf-8"))
    expected = round(sum(float(i["end"]) - float(i["start"]) for i in final["uncovered_all"]), 1)
    run = json.loads((base_run.dir / "run.json").read_text(encoding="utf-8"))
    assert run["unresolved_s"] == expected

    initial = json.loads((base_run.dir / "20_gate" / "trust.json").read_text(encoding="utf-8"))
    before = round(sum(float(i["end"]) - float(i["start"]) for i in initial["uncovered_all"]), 1)
    assert before == round(sum(end - start for start, end in EXPECTED_GAPS), 1)
    assert before > 0
    assert run["unresolved_s"] < before  # repair actually reduced them


def test_verify_ok(base_run):
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(base_run.dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["schema"] == "asr.verify/1" and report["ok"] is True
    assert report["violations"] == [] and report["checked"] > 10


def test_status_reports_run(base_run):
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "status", "--run", str(base_run.dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "partial" and payload["exit_code"] == 2
    assert [stage["name"] for stage in payload["stages"]] == STAGE_NAMES


def test_resume_recomputes_only_summary(rig, base_run, tmp_path):
    run_dir = copy_run(base_run, tmp_path)
    for name in ("summary.md", "summary.json"):
        (run_dir / "50_summary" / name).unlink()
    reset_log(rig)
    proc = pipeline(rig, run_args(rig, base_run.audio, run_dir, "--resume"))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert read_log(rig) == ["summary"]  # nothing else was spawned

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    done = [stage["name"] for stage in run["stages"] if stage["status"] == "done"]
    skipped = [stage["name"] for stage in run["stages"] if stage["status"] == "skipped"]
    assert done == ["summary", "verify"]
    assert set(skipped) == set(STAGE_NAMES) - {"summary", "verify"}
    assert all(stage.get("reason") == "inputs unchanged" for stage in run["stages"]
               if stage["status"] == "skipped")
    assert (run_dir / "50_summary" / "summary.md").exists()


@pytest.mark.parametrize("attempts", [1, 2])
@pytest.mark.parametrize("changed", ["plan", "additions"])
def test_resume_recomputes_repair_for_changed_inputs(rig, base_run, tmp_path, changed, attempts):
    env = {"ASR_STUB_ATTEMPTS": "2", "ASR_STUB_PARTIAL": "1"} if attempts == 2 else {}
    if attempts == 1:
        run_dir = copy_run(base_run, tmp_path)
    else:
        run_dir = tmp_path / "run"
        first = pipeline(rig, run_args(rig, base_run.audio, run_dir), env_extra=env)
        assert first.returncode == 2, first.stdout + first.stderr
    path = run_dir / "40_repair" / f"{changed}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if changed == "plan":
        document["total_decode_s"] += 0.25
    else:
        document["meta"] = {"changed": True}
    path.write_text(json.dumps(document), encoding="utf-8")
    reset_log(rig)
    resumed = pipeline(rig, run_args(rig, base_run.audio, run_dir, "--resume"), env_extra=env)
    assert resumed.returncode == 2, resumed.stdout + resumed.stderr
    expected_stage = "repair_run" if changed == "plan" else "repair_merge"
    assert expected_stage in read_log(rig)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["verify"]["ok"] is True
    assert run["attempts"]["repair"] == attempts
    stages = {s["name"]: s for s in run["stages"]}
    assert stages[expected_stage]["status"] == "done"
    assert all(stages[name]["status"] == "skipped" for name in ("ingest", "asr", "gate", "route"))
    verify = pipeline(rig, ["verify", "--run", run_dir])
    assert verify.returncode == 0, verify.stdout + verify.stderr
    reset_log(rig)
    again = pipeline(rig, run_args(rig, base_run.audio, run_dir, "--resume"), env_extra=env)
    assert again.returncode == 2, again.stdout + again.stderr
    assert read_log(rig) == []


@pytest.mark.parametrize("change", ["model", "command", "secondary"])
def test_resume_invalidates_engines_fingerprint(rig, audio, tmp_path, change):
    config = json.loads(rig.engines.read_text(encoding="utf-8"))
    config["secondary"] = {**config["whisper"], "model": "secondary-v1"}
    engines_path = tmp_path / "engines.json"
    engines_path.write_text(json.dumps(config), encoding="utf-8")
    local_rig = SimpleNamespace(**{**vars(rig), "engines": engines_path})
    out = tmp_path / "run"
    first = pipeline(local_rig, run_args(local_rig, audio, out))
    assert first.returncode == 2, first.stdout + first.stderr
    before = json.loads((out / "run.json").read_text(encoding="utf-8"))
    if change == "command":
        config["whisper"]["cmd"].insert(1, "-u")
    elif change == "secondary":
        config["secondary"]["model"] = "secondary-v2"
    else:
        config["whisper"]["model"] = "primary-v2"
    engines_path.write_text(json.dumps(config), encoding="utf-8")
    reset_log(local_rig)
    resumed = pipeline(local_rig, run_args(local_rig, audio, out, "--resume"))
    assert resumed.returncode == 2, resumed.stdout + resumed.stderr
    after = json.loads((out / "run.json").read_text(encoding="utf-8"))
    stages = {s["name"]: s for s in after["stages"]}
    assert stages["ingest"]["status"] == "done"
    assert stages["asr"]["status"] == "done"
    assert stages["repair_run"]["status"] == "done"
    assert after["engines_sha256"] != before["engines_sha256"]
    import hashlib
    assert after["engines_sha256"] == hashlib.sha256(engines_path.read_bytes()).hexdigest()
    assert json.loads((out / "00_input/engines.json").read_text(encoding="utf-8")) == config
    reset_log(local_rig)
    again = pipeline(local_rig, run_args(local_rig, audio, out, "--resume"))
    assert again.returncode == 2, again.stdout + again.stderr
    assert read_log(local_rig) == []


@pytest.mark.parametrize("selection", ["all", "whisper,second,broken"])
@pytest.mark.parametrize("failed_engine", ["broken", "whisper"])
def test_ensemble_degrades_to_two_engines(rig, audio, tmp_path, selection, failed_engine):
    config = json.loads(rig.engines.read_text(encoding="utf-8"))
    config["whisper"]["ensemble"] = True
    config["second"] = {**config["whisper"], "model": "second"}
    config["broken"] = {**config["whisper"], "model": "third"}
    config[failed_engine] = {"cmd": [PY, "-c", "raise SystemExit(7)"], "ensemble": True}
    config["disabled"] = {"cmd": [PY, "-c", "raise SystemExit(8)"], "ensemble": False}
    engines_path = tmp_path / "engines.json"
    engines_path.write_text(json.dumps(config), encoding="utf-8")
    stages = tmp_path / "stages"
    shutil.copytree(rig.stages, stages)
    shutil.copyfile(ROOT / "asr_ensemble.py", stages / "asr_ensemble.py")
    local_rig = SimpleNamespace(**{**vars(rig), "engines": engines_path, "stages": stages})
    out = tmp_path / "ensemble_run"
    args = run_args(local_rig, audio, out, "--ensemble", selection)
    proc = pipeline(local_rig, args)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["verify"]["ok"] is True
    ensemble = run["ensemble"]
    assert ensemble["engines"] == [name for name in ("whisper", "second", "broken") if name != failed_engine]
    assert ensemble["n_votes"] == 2
    assert ensemble["mean_confidence"] == 1.0
    report = json.loads((out / "10_asr/ensemble/report.json").read_text(encoding="utf-8"))
    assert ensemble["votes"] == report["votes"]
    assert any(failed_engine in note and "7" in note for note in run["notes"])
    assert not any("disabled" in note for note in run["notes"])
    assert all((out / "10_asr" / f"{name}.json").exists() for name in ensemble["engines"])
    segments = json.loads((out / "10_asr/segments.json").read_text(encoding="utf-8"))
    assert segments["engine"]["name"] == "ensemble"
    trust = json.loads((out / "20_gate/trust.json").read_text(encoding="utf-8"))
    assert trust["primary"] == "ensemble"
    reset_log(local_rig)
    resumed = pipeline(local_rig, [*args, "--resume"])
    assert resumed.returncode == 2, resumed.stdout + resumed.stderr
    assert read_log(local_rig) == []
    again = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert again["ensemble"] == ensemble
    assert again["notes"] == run["notes"]


@pytest.mark.parametrize("failed_primary", [False, True])
def test_ensemble_one_survivor_falls_back_without_ensemble_module(rig, audio, tmp_path, failed_primary):
    good = json.loads(rig.engines.read_text(encoding="utf-8"))["whisper"]
    bad = {"cmd": [PY, "-c", "raise SystemExit(7)"], "ensemble": True}
    config = {"whisper": bad if failed_primary else {**good, "ensemble": True},
              "second": {**good, "ensemble": True} if failed_primary else bad}
    engines = tmp_path / "engines.json"
    engines.write_text(json.dumps(config), encoding="utf-8")
    local_rig = SimpleNamespace(**{**vars(rig), "engines": engines})
    out = tmp_path / "one_survivor"
    proc = pipeline(local_rig, run_args(local_rig, audio, out, "--ensemble", "all"))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["ensemble"]["enabled"] is False
    assert run["ensemble"]["n_votes"] == 1
    assert run["ensemble"]["engines"] == (["second"] if failed_primary else ["whisper"])
    assert run["verify"]["ok"] is True
    assert any("менее 2" in note for note in run["notes"])


def test_resume_after_two_repair_attempts(rig, audio, tmp_path):
    out = tmp_path / "two_attempts"
    reset_log(rig)
    env = {"ASR_STUB_ATTEMPTS": "2", "ASR_STUB_PARTIAL": "1"}
    proc = pipeline(rig, run_args(rig, audio, out), env_extra=env)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["attempts"]["repair"] == 2
    assert run["effective_transcript"] == "40_repair/segments_merged_a2.json"
    assert (out / "40_repair" / "segments_merged_a2.json").exists()
    assert read_log(rig).count("repair_plan") == 2

    for name in ("summary.md", "summary.json"):
        (out / "50_summary" / name).unlink()
    reset_log(rig)
    resumed = pipeline(rig, run_args(rig, audio, out, "--resume"))
    assert resumed.returncode == 2, resumed.stdout + resumed.stderr
    assert read_log(rig) == ["summary"]  # the whole repair phase was reused
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    done = [stage["name"] for stage in run["stages"] if stage["status"] == "done"]
    assert done == ["summary", "verify"]
    assert run["effective_transcript"] == "40_repair/segments_merged_a2.json"
    verify = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(out)],
                            capture_output=True, text=True, cwd=str(ROOT))
    assert verify.returncode == 0, verify.stdout + verify.stderr


@pytest.mark.parametrize("relative", ["20_gate/trust.json", "21_gate_after/trust.json"])
def test_verify_catches_forged_coverage_union(base_run, tmp_path, relative):
    run_dir = copy_run(base_run, tmp_path)
    path = run_dir / relative
    trust = json.loads(path.read_text(encoding="utf-8"))
    trust["coverage_union"] = {"covered_s": 157.6, "active_s": 157.6, "coverage_union": 1.0}
    path.write_text(json.dumps(trust), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "E_COVERAGE_MISMATCH" in {v["code"] for v in json.loads(proc.stdout)["violations"]}


@pytest.mark.parametrize("damage", ["plan", "status", "pending_verify", "hash", "empty_hash", "outputs", "missing_stage", "final_trust"])
def test_verify_catches_incomplete_stage(base_run, tmp_path, damage):
    run_dir = copy_run(base_run, tmp_path)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    stage = next(s for s in run["stages"] if s["name"] == "asr")
    if damage == "missing_stage":
        run["stages"] = [s for s in run["stages"] if s["name"] != "repair_plan"]
    elif damage == "final_trust":
        run["final_trust"] = "missing/trust.json"
    elif damage == "plan":
        (run_dir / "40_repair/plan.json").unlink()
    elif damage == "pending_verify":
        next(s for s in run["stages"] if s["name"] == "verify")["status"] = "pending"
    elif damage == "status":
        stage["status"] = "failed"
    elif damage == "outputs":
        stage["outputs"] = []
    else:
        stage["inputs_hash"] = "0" * 32 if damage == "hash" else ""
    run_path.write_text(json.dumps(run), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "E_STAGE_INCOMPLETE" in {v["code"] for v in json.loads(proc.stdout)["violations"]}


@pytest.mark.parametrize("status", ["done", "skipped"])
@pytest.mark.parametrize("name,exit_code", [
    *[(name, 99) for name in STAGE_NAMES],
    ("asr", 2), ("gate", 3), ("repair_plan", 4), ("repair_run", 3), ("repair_merge", 4),
    ("asr", None), ("asr", "0"), ("asr", False), ("asr", 0.0),
])
def test_verify_catches_invalid_stage_exit(base_run, tmp_path, status, name, exit_code):
    run_dir = copy_run(base_run, tmp_path)
    path = run_dir / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    stage = next(s for s in run["stages"] if s["name"] == name)
    stage["status"] = status
    if exit_code is None:
        stage.pop("exit")
    else:
        stage["exit"] = exit_code
    path.write_text(json.dumps(run), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "E_STAGE_EXIT" in {v["code"] for v in json.loads(proc.stdout)["violations"]}


@pytest.mark.parametrize("name,exit_code", [("gate", 2), ("gate_after", 2), ("repair_plan", 2),
                                           ("repair_run", 2), ("repair_merge", 2),
                                           ("route", 3), ("route", 4), ("summary", 3), ("summary", 4)])
def test_verify_accepts_recorded_nonzero_exit_on_skipped_stage(base_run, tmp_path, name, exit_code):
    run_dir = copy_run(base_run, tmp_path)
    path = run_dir / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    stage = next(s for s in run["stages"] if s["name"] == name)
    stage.update(status="skipped", exit=exit_code)
    path.write_text(json.dumps(run), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("field,value", [("exit_code", 0), ("verdict", "ok")])
def test_verify_catches_outcome_mismatch(base_run, tmp_path, field, value):
    run_dir = copy_run(base_run, tmp_path)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run[field] = value
    run_path.write_text(json.dumps(run), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "E_EXIT_MISMATCH" in {v["code"] for v in json.loads(proc.stdout)["violations"]}


@pytest.mark.parametrize("verdict,severity", [("ok", "warn"), ("partial", "block")])
def test_verify_catches_reason_verdict_mismatch(base_run, tmp_path, verdict, severity):
    run_dir = copy_run(base_run, tmp_path)
    path = run_dir / "21_gate_after/trust.json"
    trust = json.loads(path.read_text(encoding="utf-8"))
    trust["verdict"] = verdict
    trust["reasons"][0]["severity"] = severity
    path.write_text(json.dumps(trust), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "E_EXIT_MISMATCH" in {v["code"] for v in json.loads(proc.stdout)["violations"]}


@pytest.mark.parametrize("damage", ["duration", "rms", "peak", "clip", "band", "energy", "vad", "noise",
                                    "coverage", "covered", "active", "tail", "uncovered", "dice", "lid",
                                    "guard", "empty_card", "guard_type", "malformed_card"])
def test_verify_catches_summary_mismatch(base_run, tmp_path, damage):
    run_dir = copy_run(base_run, tmp_path)
    if damage in {"guard", "empty_card", "guard_type", "malformed_card"}:
        path = run_dir / "50_summary/summary.json"
        card = json.loads(path.read_text(encoding="utf-8"))
        if damage == "malformed_card":
            card = []
        elif damage == "empty_card":
            card = {}
        elif damage == "guard_type":
            card["guard"]["phrases_ok"] = "false"
        else:
            card["guard"]["numbers_ok"] = False
        path.write_text(json.dumps(card), encoding="utf-8")
    else:
        import re
        path = run_dir / "50_summary/summary.md"
        text = path.read_text(encoding="utf-8")
        patterns = {"duration": r"(длительность: )[-\d.]+", "rms": r"(RMS )[-\d.]+",
                    "peak": r"(пик )[-\d.]+", "clip": r"(клиппинг )[-\d.]+",
                    "band": r"(доля энергии в НЧ-полосе )[-\d.]+", "energy": r"(энергия )[-\d.]+",
                    "vad": r"(VAD )[-\d.]+", "noise": r"(шумовой пол )[-\d.]+",
                    "coverage": r"(union )[-\d.]+", "covered": r"(covered )[-\d.]+",
                    "active": r"(с из )[-\d.]+", "tail": r"(хвост без текста )[-\d.]+",
                    "uncovered": r"(uncovered_s: )[-\d.]+", "dice": r"(среднее dice )[-\d.]+",
                    "lid": r"(вероятность )[-\d.]+"}
        text, count = re.subn(patterns[damage], r"\g<1>999.99", text)
        assert count == 1
        path.write_text(text, encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "E_SUMMARY_MISMATCH" in {v["code"] for v in json.loads(proc.stdout)["violations"]}


@pytest.mark.parametrize("verdict,exit_code", [("failed", 3), ("blocked", 4)])
def test_verify_forbids_summary_for_run_outcome(base_run, tmp_path, verdict, exit_code):
    run_dir = copy_run(base_run, tmp_path)
    path = run_dir / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run.update(verdict=verdict, exit_code=exit_code)
    path.write_text(json.dumps(run), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "E_SUMMARY_FORBIDDEN" in {v["code"] for v in json.loads(proc.stdout)["violations"]}


def test_verify_catches_broken_trust(base_run, tmp_path):
    run_dir = copy_run(base_run, tmp_path)
    path = run_dir / "21_gate_after" / "trust.json"
    trust = json.loads(path.read_text(encoding="utf-8"))
    trust["uncovered_all"] = [{"start": 100.0, "end": 130.0}]  # 30 s the run.json never saw
    path.write_text(json.dumps(trust, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3
    report = json.loads(proc.stdout)
    assert report["ok"] is False
    codes = {violation["code"] for violation in report["violations"]}
    assert "E_UNRESOLVED_MISMATCH" in codes


def test_verify_catches_broken_segments(base_run, tmp_path):
    run_dir = copy_run(base_run, tmp_path)
    path = run_dir / "40_repair" / "segments_merged.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["segments"][3]["start"] = 0.5
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3
    codes = {violation["code"] for violation in json.loads(proc.stdout)["violations"]}
    assert "E_SEGMENTS_ORDER" in codes


def test_verify_catches_missing_artifact(base_run, tmp_path):
    run_dir = copy_run(base_run, tmp_path)
    (run_dir / "30_route" / "route.json").unlink()
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(run_dir)],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 3
    codes = {violation["code"] for violation in json.loads(proc.stdout)["violations"]}
    assert "E_MISSING_ROUTE" in codes


def test_blocked_route_exit_4_and_summary_forbidden(rig, audio, tmp_path):
    out = tmp_path / "blocked"
    reset_log(rig)
    proc = pipeline(rig, run_args(rig, audio, out), env_extra={"ASR_STUB_ROUTE": "blocked"})
    assert proc.returncode == 4, proc.stdout + proc.stderr
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["verdict"] == "blocked" and run["exit_code"] == 4
    statuses = {stage["name"]: stage["status"] for stage in run["stages"]}
    assert statuses["repair"] == "skipped" and statuses["summary"] == "skipped"
    assert not (out / "50_summary" / "summary.md").exists()

    verify = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(out)],
                            capture_output=True, text=True, cwd=str(ROOT))
    assert verify.returncode == 0, verify.stdout + verify.stderr

    (out / "50_summary").mkdir(exist_ok=True)
    (out / "50_summary" / "summary.md").write_text("# поздний summary\n", encoding="utf-8")
    verify = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(out)],
                            capture_output=True, text=True, cwd=str(ROOT))
    assert verify.returncode == 3
    codes = {violation["code"] for violation in json.loads(verify.stdout)["violations"]}
    assert "E_SUMMARY_FORBIDDEN" in codes

    legacy = pipeline(rig, run_args(rig, audio, out, "--legacy-exit"),
                      env_extra={"ASR_STUB_ROUTE": "blocked"})
    assert legacy.returncode == 2
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["exit_code"] == 4


def test_failed_verdict_exit_3_without_summary(rig, audio, tmp_path):
    out = tmp_path / "failed"
    reset_log(rig)
    proc = pipeline(rig, run_args(rig, audio, out), env_extra={"ASR_STUB_VERDICT": "failed"})
    assert proc.returncode == 3, proc.stdout + proc.stderr
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["verdict"] == "failed" and run["exit_code"] == 3
    assert not (out / "50_summary" / "summary.md").exists()
    verify = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(out)],
                            capture_output=True, text=True, cwd=str(ROOT))
    assert verify.returncode == 0, verify.stdout + verify.stderr

    legacy = pipeline(rig, run_args(rig, audio, out, "--legacy-exit"),
                      env_extra={"ASR_STUB_VERDICT": "failed"})
    assert legacy.returncode == 2


def test_refused_summary_is_diagnostic_only_and_resumable(rig, audio, tmp_path):
    out = tmp_path / "refused"
    args = run_args(rig, audio, out)
    env = {"ASR_STUB_REFUSE": "1"}
    proc = pipeline(rig, args, env_extra=env)
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert not (out / "50_summary/summary.md").exists()
    assert not (out / "50_summary/summary.json").exists()
    assert (out / "50_summary/summary.rejected.json").exists()
    verify = pipeline(rig, ["verify", "--run", out])
    assert verify.returncode == 0, verify.stdout + verify.stderr
    reset_log(rig)
    resumed = pipeline(rig, [*args, "--resume"], env_extra=env)
    assert resumed.returncode == 4, resumed.stdout + resumed.stderr
    assert read_log(rig) == []


def test_missing_stage_module_is_internal_error(rig, audio, tmp_path):
    stages = tmp_path / "partial_stages"
    stages.mkdir()
    for name in ("asr_gate.py", "adapt_legacy.py"):
        source = rig.stages / name
        shutil.copyfile(source, stages / name)
    out = tmp_path / "run"
    proc = pipeline(rig, ["run", "--audio", str(audio), "--out", str(out), "--engines", str(rig.engines),
                          "--stages-dir", str(stages)])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["error"]["code"] == "E_MODULE_MISSING"
    assert "asr_router.py" in payload["error"]["detail"]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert {stage["name"]: stage["status"] for stage in run["stages"]}["route"] == "failed"


def test_bad_arguments_exit_64(rig, tmp_path):
    proc = pipeline(rig, ["run", "--out", str(tmp_path / "no_audio")])
    assert proc.returncode == 64
    payload = json.loads(proc.stdout)
    assert payload["error"]["code"] == "E_ARGS" and payload["exit_code"] == 64

    proc = pipeline(rig, ["run", "--audio", str(tmp_path / "missing.wav"), "--out", str(tmp_path / "x"),
                          "--engines", str(rig.engines), "--stages-dir", str(rig.stages)])
    assert proc.returncode == 64
    assert json.loads(proc.stdout)["error"]["code"] == "E_AUDIO"


# ------------------------------------------------------------- adapt_legacy

def test_convert_flat_list():
    doc = AL.convert([{"start": 0, "end": 1.5, "text": "  привет   мир "},
                      {"start": 2.0, "end": 3.0, "text": "ещё"}])
    assert doc["schema"] == "asr.segments/1"
    assert [row["id"] for row in doc["segments"]] == ["s0001", "s0002"]
    assert doc["segments"][0]["text"] == "привет мир"
    assert doc["segments"][0]["start"] == 0.0 and doc["segments"][0]["end"] == 1.5
    assert doc["segments"][0]["flags"] == [] and doc["segments"][0]["speaker"] is None
    assert doc["meta"]["source_format"] == "flat"


def test_convert_nested_and_envelope():
    doc = AL.convert({"segments": [{"start": 0, "end": 1, "text": "а"}],
                      "audio": {"path": "/x.wav", "duration_s": 10, "sr": 16000, "channels": 1}})
    assert doc["meta"]["source_format"] == "nested"
    assert doc["audio"]["duration_s"] == 10 and doc["audio"]["sr"] == 16000

    again = AL.convert(doc)
    assert again["schema"] == "asr.segments/1"
    assert again["segments"][0]["id"] == "s0001"
    assert again["audio"]["path"] == "/x.wav"


def test_convert_x20_units():
    records = [
        {"id": "u1", "unit": {"start": 1.0, "end": 2.5}, "status": "ok",
         "result": {"text": "привет"}},
        {"id": "u2", "unit": "00:03.000-00:05.500", "status": "done",
         "result": {"text": "мир"}},
        {"id": "u3", "unit": [7, 9], "status": "error", "result": {"text": "ошибка"}},
    ]
    doc = AL.convert(records, audio={"duration_s": 20})
    assert doc["meta"]["source_format"] == "x20"
    assert [(row["start"], row["end"]) for row in doc["segments"]] == [(1.0, 2.5), (3.0, 5.5), (7.0, 9.0)]
    assert doc["segments"][0]["text"] == "привет"
    assert doc["segments"][2]["flags"] == ["legacy_status:error"]


def test_convert_x20_without_timing_raises():
    with pytest.raises(AL.AdaptError) as excinfo:
        AL.convert([{"id": "u", "status": "ok", "result": {"text": "без границ"}}])
    assert excinfo.value.code == "E_NO_TIMING"


def test_convert_clamps_and_skips_invalid():
    doc = AL.convert([{"start": 0, "end": 12, "text": "длинный"},
                      {"start": 5, "end": 5, "text": "нулевой"},
                      {"start": 20, "end": 21, "text": "поздний"},
                      {"start": 2, "end": 3, "text": "   "}],
                     audio={"duration_s": 10})
    assert [row["text"] for row in doc["segments"]] == ["длинный"]
    assert doc["segments"][0]["end"] == 10.0
    assert doc["meta"]["clamped"] == 1
    assert doc["meta"]["skipped"] == 3


def test_convert_cli(tmp_path):
    src = tmp_path / "legacy.json"
    src.write_text(json.dumps([{"start": 0, "end": 1, "text": "раз"}]), encoding="utf-8")
    dst = tmp_path / "segments.json"
    proc = subprocess.run([PY, str(ROOT / "adapt_legacy.py"), "--in", str(src), "--out", str(dst),
                           "--duration", "10", "--engine", "legacy-test"],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["segments"] == 1 and payload["format"] == "flat"
    doc = json.loads(dst.read_text(encoding="utf-8"))
    assert doc["schema"] == "asr.segments/1" and doc["audio"]["duration_s"] == 10
    assert doc["engine"]["name"] == "legacy-test"


def test_convert_cli_errors(tmp_path):
    src = tmp_path / "bad.json"
    src.write_text(json.dumps([{"unit": None, "result": {"text": "x"}}]), encoding="utf-8")
    proc = subprocess.run([PY, str(ROOT / "adapt_legacy.py"), "--in", str(src),
                           "--out", str(tmp_path / "out.json")],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["error"]["code"] == "E_NO_TIMING"


# ------------------------------------------------- real modules (may be absent)

@pytest.fixture(scope="session")
def tone_audio(tmp_path_factory):
    """180 s of tone bursts over the whisper intervals (+1 burst in a hole).

    Speech-like energy lets the real gate/router/repair/summary run without any
    ML model: `--vad energy --lid declared`.
    """
    np = pytest.importorskip("numpy")
    sr = 16000
    total = int(DURATION * sr)
    samples = np.zeros(total, dtype=np.float32)
    axis = np.arange(total, dtype=np.float32) / sr
    tone = (0.06 * np.sin(2 * np.pi * 220 * axis)).astype(np.float32)
    for start, end in [(float(row["start"]), float(row["end"])) for row in WHISPER] + [(96.2, 98.0)]:
        lo, hi = int(start * sr), min(int(end * sr), total)
        samples[lo:hi] = tone[lo:hi]
    path = tmp_path_factory.mktemp("tone") / "tone.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture(scope="session")
def real_run(rig, tone_audio, tmp_path_factory):
    """Real stage modules + stub engine, no ML flags."""
    missing = [name for name in STAGE_FLAGS if not (ROOT / name).exists()]
    if missing:
        pytest.skip(f"нет реальных модулей стадий {missing}")
    out = tmp_path_factory.mktemp("real_run") / "run"
    reset_log(rig)
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "run", "--audio", str(tone_audio),
                           "--out", str(out), "--engines", str(rig.engines),
                           "--peer", f"kts={KTS}", "--vad", "energy", "--lid", "declared"],
                          capture_output=True, text=True, env=stage_env(rig),
                          cwd=str(ROOT), timeout=900)
    return SimpleNamespace(dir=out, proc=proc)


def test_real_stage_modules_end_to_end(real_run):
    assert real_run.proc.returncode in (0, 2, 4), real_run.proc.stdout + real_run.proc.stderr
    run = json.loads((real_run.dir / "run.json").read_text(encoding="utf-8"))
    names = [stage["name"] for stage in run["stages"]]
    assert names[:4] == ["ingest", "asr", "gate", "route"]
    assert names[-1] == "verify"
    assert all(stage["status"] in {"done", "skipped"} for stage in run["stages"])
    assert run["unresolved_s"] is not None

    if real_run.proc.returncode == 4:
        assert run.get("summary_refused") or run.get("verdict") == "blocked"

    summary_md = real_run.dir / "50_summary" / "summary.md"
    if run["verdict"] == "failed" or run["verdict"] == "blocked" or run.get("summary_refused"):
        assert not summary_md.exists()
    else:
        assert summary_md.exists()

    verify = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "verify", "--run", str(real_run.dir)],
                            capture_output=True, text=True, cwd=str(ROOT))
    assert verify.returncode == 0, verify.stdout + verify.stderr


STAGE_FLAGS = {
    "asr_gate.py": ("--primary", "--vad", "--lid"),
    "asr_router.py": ("--trust", "--out"),
    "asr_repair.py": ("plan", "run", "merge"),
    "asr_summary.py": ("--trust", "--transcript"),
}


@pytest.mark.parametrize("module", sorted(STAGE_FLAGS))
def test_real_module_matches_contract_cli(module):
    """Probe the real module's CLI only; skipped while it is being written."""
    path = ROOT / module
    if not path.exists():
        pytest.skip(f"{module} ещё не реализован — контрактная проверка отложена")
    proc = subprocess.run([PY, str(path), "--help"], capture_output=True, text=True, timeout=120,
                          cwd=str(ROOT))
    text = (proc.stdout or "") + (proc.stderr or "")
    missing = [flag for flag in STAGE_FLAGS[module] if flag not in text]
    if missing:
        pytest.skip(f"{module} не поддерживает контрактные флаги {missing} — модуль ещё в работе")
    assert proc.returncode == 0


@pytest.mark.skipif(os.environ.get("ASR_RUN_INTEGRATION") != "1",
                    reason="полный прогон с реальными модулями требует ML (ASR_RUN_INTEGRATION=1)")
def test_real_pipeline_smoke(tmp_path):
    missing = [name for name in ("asr_gate.py", "asr_router.py", "asr_repair.py", "asr_summary.py")
               if not (ROOT / name).exists()]
    if missing:
        pytest.skip(f"нет модулей {missing}")
    raw_audio = os.environ.get("ASR_TEST_AUDIO")
    if not raw_audio:
        pytest.skip("ASR_TEST_AUDIO не задан")
    fixture_audio = Path(raw_audio)
    if not fixture_audio.exists():
        pytest.skip("нет тестового аудио")
    proc = subprocess.run([PY, str(ROOT / "asr_pipeline.py"), "run", "--audio", str(fixture_audio),
                           "--out", str(tmp_path / "run"), "--peer", f"kts={KTS}"],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=3600)
    assert proc.returncode in (0, 2, 3), proc.stdout + proc.stderr
