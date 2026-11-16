#!/usr/bin/env python3
"""X20 ensemble engines: gigaam | whisper | parakeet | qwen.

Runs under the interpreter that matches the model family (see engines.json):
  gigaam            -> x20/work/ru_asr/venv/bin/python
  whisper/parakeet/qwen -> x20/work/venv/bin/python

Reads a wav, slices it into non-overlapping windows, transcribes each window
and writes `asr.segments/1` with global timecodes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

def resolve_x20_root() -> Path:
    """One canonical env source; reuse the shipped config default for direct CLI."""
    value = os.environ.get("ASR_X20_ROOT") or os.environ.get("X20_ROOT")  # legacy alias accepted
    if not value:
        source = Path(__file__).resolve().parents[1] / "engines.json"
        try:
            config = json.loads(source.read_text(encoding="utf-8"))
            engines = config.get("engines", config)
            defaults = set()
            for cfg in engines.values():
                if not isinstance(cfg, dict):
                    continue
                cmd = cfg.get("cmd", [])
                parts = cmd if isinstance(cmd, list) else [cmd]
                for part in parts:
                    defaults.update(re.findall(r"\$\{ASR_X20_ROOT:-([^}]+)\}", part))
            if len(defaults) == 1:
                value = defaults.pop()
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    if not value:
        raise SystemExit("Задайте ASR_X20_ROOT: нет однозначного default в engines.json")
    resolved = Path(value).expanduser().resolve()
    os.environ["ASR_X20_ROOT"] = str(resolved)
    os.environ["X20_ROOT"] = str(resolved)  # output alias only, never an input
    return resolved


def venv_site_packages(venv: Path) -> list[Path]:
    """Find the sibling venv packages compatible with the running interpreter."""
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return sorted(path for path in (venv / "lib").glob("python*/site-packages")
                  if path.is_dir() and path.parent.name == version)


X20 = resolve_x20_root()
MODELS = X20 / "work" / "models"


def log(msg: str) -> None:
    print(f"[x20:{os.environ.get('X20_ENGINE', '?')}] {msg}", file=sys.stderr, flush=True)


def load(engine: str):
    if engine == "gigaam":
        p = MODELS / "gigaam_hf"
        sys.path.insert(0, str(p))
        # Borrow transformers from the sibling MLX venv for this Python ABI.
        packages = venv_site_packages(X20 / "work" / "venv")
        if not packages:
            raise SystemExit("Нет совместимого site-packages в x20 work/venv; проверьте ASR_X20_ROOT и версии Python")
        for path in packages:
            if str(path) not in sys.path:
                sys.path.append(str(path))
        import torch
        import modeling_gigaam as impl
        cfg = json.loads((p / "config.json").read_text())
        model = impl.GigaAMModel(impl.GigaAMConfig(**cfg))
        weights = torch.load(p / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        del weights
        model.model.encoder = model.model.encoder.half()
        model = model.eval().to("mps")

        def run(x, sr):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                sf.write(fh.name, x, sr)
                path = fh.name
            try:
                with torch.inference_mode():
                    text = model.transcribe(path)
                torch.mps.synchronize()
            finally:
                os.unlink(path)
            return str(text)

        return run

    import mlx.core as mx

    if engine == "whisper":
        import mlx_whisper

        def run(x, sr):
            try:
                out = mlx_whisper.transcribe(x, path_or_hf_repo=str(MODELS / "whisper"), language="ru",
                                             temperature=0, condition_on_previous_text=False, word_timestamps=False,
                                             verbose=None, fp16=True, no_speech_threshold=0.6)
                return str(out.get("text") or "")
            finally:
                mx.clear_cache()

        return run

    if engine == "parakeet":
        import dataclasses
        from mlx_audio.stt.models.parakeet.parakeet import Model
        model = Model.from_pretrained(str(MODELS / "parakeet"), dtype=mx.float32)

        def run(x, sr):
            try:
                res = dataclasses.asdict(model.generate(mx.array(x), verbose=False,
                                                        chunk_duration=30.0, overlap_duration=3.0))
                return str(res.get("text") or "")
            finally:
                mx.clear_cache()

        return run

    if engine == "qwen":
        from mlx_audio.stt import load as load_model
        model = load_model(str(MODELS / "qwen3_asr_4bit"))

        def run(x, sr):
            import dataclasses
            try:
                res = dataclasses.asdict(model.generate(mx.array(x), language="Russian", temperature=0,
                                                        max_tokens=256, repetition_penalty=1.1, verbose=False))
                return str(res.get("text") or "")
            finally:
                mx.clear_cache()

        return run

    raise SystemExit(f"unknown engine {engine}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["gigaam", "whisper", "parakeet", "qwen"])
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="ru")
    ap.add_argument("--window", type=float, default=20.0)
    a = ap.parse_args()
    os.environ["X20_ENGINE"] = a.engine

    x, sr = sf.read(a.audio, dtype="float32", always_2d=False)
    if sr != 16000:
        raise SystemExit(f"expected 16000 Hz audio, got {sr} Hz; resample with ffmpeg -ar 16000 first")
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    gain = (0.891 / peak) if peak > 1e-6 else 1.0
    if gain != 1.0:
        x = x * gain
        log(f"peak-normalised x{gain:.2f}")
    dur = len(x) / sr

    t0 = time.time()
    run = load(a.engine)
    load_s = time.time() - t0
    segments, t_decode = [], 0.0
    n = int(math.ceil(dur / a.window))
    for i in range(n):
        start, end = i * a.window, min((i + 1) * a.window, dur)
        clip = x[int(start * sr):int(end * sr)]
        if len(clip) < int(0.5 * sr):
            continue
        t = time.time()
        text = run(clip, sr).strip()
        t_decode += time.time() - t
        if text:
            segments.append({"id": f"s{len(segments) + 1:04d}", "start": round(start, 2),
                             "end": round(end, 2), "text": text, "speaker": None, "conf": None,
                             "source": a.engine, "flags": []})
    doc = {"schema": "asr.segments/1",
           "audio": {"path": str(Path(a.audio).resolve()), "sha256": "", "duration_s": round(dur, 2),
                     "sr": sr, "channels": 1},
           "engine": {"name": a.engine, "model": a.engine, "version": "x20", "params": {"window_s": a.window,
                                                                                       "input_gain": round(gain, 3)},
                      "level_sensitive": True},
           "segments": segments, "meta": {"vad": "off", "window_s": a.window,
                                          "load_s": round(load_s, 1), "decode_s": round(t_decode, 1)}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{len(segments)} segments, load {load_s:.1f}s, decode {t_decode:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
