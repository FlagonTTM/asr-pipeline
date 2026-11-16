#!/usr/bin/env bash
# Offline orchestration smoke: real fixture transcripts, synthetic audio, no ML.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Нет .venv/bin/python. Подготовьте окружение по README.md.\n' >&2
  exit 64
fi
# Caller controls scratch via TMPDIR; /tmp is the portable standalone fallback.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/asr-smoke.XXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON" - "$WORK/audio.wav" <<'PY'
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
sr = 16000
# Duration matches the checked-in whisper/kts fixtures, not a real recording.
t = np.arange(int(178.25 * sr), dtype=np.float32) / sr
sf.write(str(Path(sys.argv[1])), 0.25 * np.sin(2 * np.pi * 440 * t), sr)
PY
expect_partial() {
  local label="$1" code=0
  shift
  "$@" >"$WORK/$label.log" 2>&1 || code=$?
  if [[ "$code" != 2 ]]; then
    printf 'FAIL %s: ожидался exit 2, получен %s\n' "$label" "$code" >&2
    "$PYTHON" - "$WORK/$label.log" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).read_text(encoding="utf-8"), file=sys.stderr)
PY
    exit 1
  fi
  printf 'PASS %s: exit 2\n' "$label"
}
expect_partial gate "$PYTHON" "$ROOT/asr_pipeline.py" gate \
  --audio "$WORK/audio.wav" --segments "$ROOT/fixtures/whisper.json" \
  --peer "kts=$ROOT/fixtures/kts.json" --primary whisper \
  --vad off --lid declared --language ru --out "$WORK/20_gate"
expect_partial route "$PYTHON" "$ROOT/asr_pipeline.py" route \
  --trust "$WORK/20_gate/trust.json" --out "$WORK/30_route/route.json"
expect_partial summary "$PYTHON" "$ROOT/asr_pipeline.py" summarize \
  --trust "$WORK/20_gate/trust.json" --transcript "$ROOT/fixtures/whisper.json" \
  --out "$WORK/50_summary"
"$PYTHON" - "$WORK" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
trust = json.loads((root / "20_gate/trust.json").read_text(encoding="utf-8"))
route = json.loads((root / "30_route/route.json").read_text(encoding="utf-8"))
card = json.loads((root / "50_summary/summary.json").read_text(encoding="utf-8"))
markdown = (root / "50_summary/summary.md").read_text(encoding="utf-8")
assert trust["schema"] == "asr.trust/2" and trust["verdict"] == "partial"
assert trust["primary"] == "whisper" and set(trust["coverage"]) == {"whisper", "kts"}
assert trust["language_id"]["method"] == "declared"
assert trust["activity"]["vad_active_s"] is None
assert trust["uncovered_all"] and 0 < trust["coverage_union"]["coverage_union"] < 1
assert route["schema"] == "asr.route/1" and route["summary_allowed"] is True
assert route["blocked"] is None
assert card["schema"] == "asr.summary/1" and card["status"] == "ok"
assert card["trusted"] is False and not card["guard"]["violations"]
assert "⚠" in markdown and "## Качество записи" in markdown
print("PASS artifacts: trust/route/summary schema, coverage, guard and caveat")
PY
printf 'SMOKE PASS: без моделей, сети и внешнего аудио; временные артефакты удалены при выходе.\n'
