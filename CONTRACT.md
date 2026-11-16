# Контракт пайплайна ASR (общий для всех модулей)

Проект: `/Users/Ivan/PycharmProjects/asr-pipeline`.
Интерпретатор для запуска и тестов: `/Users/Ivan/PycharmProjects/asr-pipeline/.venv/bin/python`.
В нём уже стоят: faster-whisper, soundfile, numpy<2, pytest. **Ничего не устанавливать, в сеть не ходить.**

Правила для всех модулей:

- только stdlib + numpy/soundfile (+ faster-whisper там, где нужен LID/VAD);
- никаких импортов между новыми модулями, кроме `asr_gate` (его публичные функции можно переиспользовать);
- каждый модуль запускается и как CLI, и импортируется; тесты — `tests/test_<module>.py`, запуск `pytest -q`, без сети и без ML (кроме явно помеченных);
- пути только абсолютные либо от `Path(__file__).parent`;
- ошибки — исключения с кодом в тексте; CLI печатает JSON в stdout и завершаетcя кодом.

## Схемы

### segments (`asr.segments/1`)

```json
{"schema":"asr.segments/1",
 "audio":{"path":"/abs/audio.wav","sha256":"...","duration_s":178.25,"sr":16000,"channels":1},
 "engine":{"name":"whisper","model":"large-v3","version":"ct2","params":{},"level_sensitive":true},
 "segments":[{"id":"s0001","start":0.0,"end":7.5,"text":"...","speaker":null,"conf":null,"source":"primary","flags":[]}],
 "meta":{"vad":"silero","window_s":null}}
```

Инварианты: `start < end`, монотонность по `start`, `end <= duration_s + 0.1`, `text` непустой.

### trust (`asr.trust/2`)

```json
{"schema":"asr.trust/2","verdict":"ok|partial|failed","primary":"whisper","gate_version":"2.0",
 "reasons":[{"code":"SPEECH_GAP","severity":"warn|block","detail":"...","intervals":[{"start":1.0,"end":2.0}]}],
 "audio_metrics":{"duration_s":178.25,"rms_dbfs":-41.58,"peak_dbfs":-13.09,"clip_fraction":0.0,"band_share_below_1khz":0.751,"rms_per_10s":[]},
 "activity":{"energy_active_s":157.6,"vad_active_s":31.2,"noise_floor_dbfs":-83.8},
 "language_id":{"language":"ru","probability":0.971,"method":"whisper|declared"},
 "coverage":{"whisper":{"coverage":0.87,"coverage_union":0.87,"covered_s":150.0,"union_active_s":157.6,"tail_uncovered_s":7.1,"speech_gaps":[{"start":1,"end":2}]}},
 "coverage_union":{"covered_s":150.0,"active_s":157.6,"coverage_union":0.87},
 "uncovered_all":[{"start":1.0,"end":2.0}],
 "agreement":{"mean_dice":0.28,"windows":[{"start":0,"end":20,"dice":0.3}],"pair":["whisper","kts"]},
 "config":{"vad":"energy","lid":"whisper","thresholds":{}},
 "verdict_legacy":"partial"}
```

Коды причин: `LANGUAGE_MISMATCH`, `LOW_LEVEL`, `CLIPPING`, `VAD_UNRELIABLE`, `TAIL_UNCOVERED`, `LOW_COVERAGE`, `SPEECH_GAP`, `LOW_AGREEMENT`. `severity` обязателен; `block` → `failed`, иначе `partial`.

Покрытие считается **только union** (объединение интервалов сегментов), знаменатель — энергетическая активность. Дыры — интервалы активности, не покрытые union, длиной ≥ 1.0 с (не «вложенность сегмента»).

### route (`asr.route/1`)

```json
{"schema":"asr.route/1","verdict":"partial","codes":["SPEECH_GAP"],"blocked":null,
 "steps":[{"id":"repair_decode","kind":"repair","because":"SPEECH_GAP","params":{"intervals":[{"start":40.2,"end":54.9}]}}],
 "budget":{"attempts_max":2,"seconds_max":62.4},
 "summary_allowed":true,"summary_mode":"caveated","unresolved":[],"notes":[]}
```

Маппинг: `LOW_LEVEL`→normalize; `CLIPPING`→no_gain (+block при clip_fraction>0.005); `VAD_UNRELIABLE`→detector=energy; `LOW_COVERAGE`→second_engine (+block после повтора при union<0.65); `TAIL_UNCOVERED`/`SPEECH_GAP`→repair_decode; `LOW_AGREEMENT`→arbitrate; `LANGUAGE_MISMATCH`→blocked, summary_allowed=false; неизвестный код→blocked `UNKNOWN_REASON`.

Бюджет ремонта: `seconds_max = round(0.35 * duration_s)`.

### repair (`asr.repair.plan/1`)

```json
{"schema":"asr.repair.plan/1","attempt":1,"intervals":[{"id":"r1","start":40.23,"end":54.92,"pad_s":0.3,"reason":"gap","engines":["whisper"],"decode_s":15.0}],
 "total_decode_s":60.0,"budget_s":62.4,"within_budget":true,"stop_rule":{"min_gain_ratio":0.25,"max_attempts":2}}
```

Ремонт режет **только** union-дыры ≥1.0 с, паддинг 0.3 с, фейды 50 мс, склейка по границам дыры (`clipped=true`), инварианты M1–M6: нет перекрытий >0.2 с; добавления внутри планового интервала ±0.05 с; нет повтора 6-граммы в пределах 2 с; union-покрытие растёт; сегментов ≤ 2× базы; таймкоды монотонны.

### summary (`asr.summary/1`)

```json
{"trusted":false,"verdict":"partial","codes":["SPEECH_GAP"],
 "guard":{"numbers_ok":true,"tasks_quoted":true,"caveat_present":true},"numbers_sha256":"..."}
```

`summary.md`: `## Качество записи` (числа рендерит код, байт-стабильно), `## О чём встреча`, `## Решения`, `## Задачи`, `## Спорные места`. При `partial` первая строка после заголовка — `⚠` и коды. Guard: каждое число в теле обязано быть в числах trust.json (±0.01) или таймкодом; под задачей — цитата ≤120 символов из транскрипта; запрещены «точность N%».

### run.json (`asr.run/1`)

```json
{"schema":"asr.run/1","stages":[{"name":"gate","status":"done","inputs_hash":"...","outputs":["20_gate/trust.json"],"exit":0}],
 "attempts":{"prep":0,"repair":0},"engines":{"whisper":{"cmd":"..."}},"verdict":"partial","exit_code":2,"unresolved_s":7.1}
```

## CLI

| Модуль | Команда |
|---|---|
| `asr_gate.py` | `--audio --transcript NAME=PATH [--transcript ...] [--evidence NAME=PATH] --primary NAME --vad {silero,energy,off} --lid {whisper,declared,off} --language ru --out DIR` |
| `asr_router.py` | `--trust PATH --out PATH [--budget-frac 0.35]` |
| `asr_repair.py` | `plan --trust PATH --route PATH --out DIR`; `run --audio PATH --plan PATH --engines engines.json --out DIR`; `merge --base PATH --additions PATH --out PATH` |
| `asr_summary.py` | `--trust PATH --transcript PATH [--segments PATH] --out DIR` |
| `asr_pipeline.py` | `run --audio PATH --out DIR [--engines engines.json] [--peer NAME=segments.json] [--resume]`; подкоманды `gate route repair summarize verify status` |

## Exit-коды

`0` ok; `2` partial; `3` failed; `4` blocked (нужен человек); `1` внутренняя ошибка; `64` неверные аргументы. `asr_gate.py` — только 0/2.

## Тестовые данные (фикстуры)

- `/Users/Ivan/.hermes/attachments/audio.wav` — 178.25 с, 16 кГц моно, RMS −41.58 dBFS.
- `fixtures/whisper.json` — сегменты Whisper large-v3 с этого аудио (16 сегментов).
- `fixtures/kts.json` — сегменты стенда КТС (10 сегментов, дыра 1:56–2:28).
- `fixtures/docx_plain.txt` — текст docx-ансамбля.

## Владение файлами

Каждый агент правит только свои файлы; общие (`CONTRACT.md`, `fixtures/`, `engines.json`) не трогает. Конфликт владения = отказ от правки.
