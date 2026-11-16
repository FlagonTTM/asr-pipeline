#!/usr/bin/env python3
"""Linux engines: gigaam | parakeet | qwen via transformers (CPU/CUDA).

Same contract as engines/x20_engine.py: 20 s windows, peak normalisation to
-1 dBFS, `asr.segments/1` out. Weights are read offline from HF_HOME.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def log(msg: str) -> None:
    print(f"[linux:{os.environ.get('ASR_ENGINE', '?')}] {msg}", file=sys.stderr, flush=True)


def load(engine: str):
    import torch
    from transformers import AutoModel, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if engine == "gigaam":
        model = AutoModel.from_pretrained("ai-sage/GigaAM-v3", trust_remote_code=True,
                                          local_files_only=True).eval().to(device)

        def run(x, sr):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                sf.write(fh.name, x, sr)
                path = fh.name
            try:
                with torch.inference_mode():
                    out = model.transcribe(path)
            finally:
                os.unlink(path)
            return str(out)

        return run

    if engine == "parakeet":
        processor = AutoProcessor.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", local_files_only=True)
        model = AutoModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", dtype=dtype,
                                          local_files_only=True).eval().to(device)

        def run(x, sr):
            inputs = processor(x, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.inference_mode():
                ids = model.generate(**inputs)
            return processor.batch_decode(ids, skip_special_tokens=True)[0]

        return run

    if engine == "qwen":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B", local_files_only=True)
        model = AutoModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=dtype,
                                          local_files_only=True).eval().to(device)

        def run(x, sr):
            inputs = processor(x, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.inference_mode():
                ids = model.generate(**inputs)
            return processor.batch_decode(ids, skip_special_tokens=True)[0]

        return run

    raise SystemExit(f"unknown engine {engine}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["gigaam", "parakeet", "qwen"])
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="ru")
    ap.add_argument("--window", type=float, default=20.0)
    a = ap.parse_args()
    os.environ["ASR_ENGINE"] = a.engine

    x, sr = sf.read(a.audio, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"нужен 16 кГц, получено {sr}")
    peak = float(np.abs(x).max()) if len(x) else 0.0
    gain = (0.891 / peak) if peak > 1e-6 else 1.0
    x = x * gain
    dur = len(x) / sr

    t0 = time.time()
    run = load(a.engine)
    load_s = time.time() - t0
    segments, decode_s = [], 0.0
    for i in range(int(math.ceil(dur / a.window))):
        start, end = i * a.window, min((i + 1) * a.window, dur)
        clip = x[int(start * sr):int(end * sr)]
        if len(clip) < int(0.5 * sr):
            continue
        t = time.time()
        text = run(clip, sr).strip()
        decode_s += time.time() - t
        if text:
            segments.append({"id": f"s{len(segments) + 1:04d}", "start": round(start, 2),
                             "end": round(end, 2), "text": text, "speaker": None, "conf": None,
                             "source": a.engine, "flags": []})
    doc = {"schema": "asr.segments/1",
           "audio": {"path": str(Path(a.audio).resolve()), "sha256": "", "duration_s": round(dur, 2),
                     "sr": sr, "channels": 1},
           "engine": {"name": a.engine, "model": a.engine, "version": "linux-transformers",
                      "params": {"window_s": a.window, "input_gain": round(gain, 3)}, "level_sensitive": True},
           "segments": segments,
           "meta": {"vad": "off", "window_s": a.window, "load_s": round(load_s, 1),
                    "decode_s": round(decode_s, 1)}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{len(segments)} segments, load {load_s:.1f}s, decode {decode_s:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
