# Интеграция ASR Pipeline

## Точки входа

Публичная граница — отдельный процесс `asr_pipeline.py`, не импорт внутренних
классов. Запускайте Python **из `.venv` этой поставки** и задавайте абсолютные
пути в своём оркестраторе. `run.sh` — совместимая обёртка полного run с
разворачиванием env-шаблонов движков; быстрый старт и состав — в `README.md`.

| CLI после `.venv/bin/python asr_pipeline.py` | Вход → выход |
|---|---|
| `run --audio WAV --out DIR [--engines JSON] [--engine NAME] [--ensemble all] [--peer NAME=JSON] [--resume]` | Полный проход; run.json, артефакты стадий, JSON `asr.run.report/1` в stdout |
| `gate --audio WAV --segments JSON [--peer NAME=JSON] --primary NAME --vad energy --lid declared --out DIR` | Готовые гипотезы → trust.json/trust.md/vocab.md; модели ASR не запускаются |
| `route --trust JSON --out PATH [--budget-frac 0.35]` | Карточка доверия → route.json |
| `repair --mode plan --trust JSON --route JSON --out DIR` | Union-дыры → plan.json |
| `repair --mode run --audio WAV --plan JSON --engines JSON --out DIR` | План + движки → additions.json, attempt-отчёты, cuts/engine |
| `repair --mode merge --base JSON --additions JSON --out FILE` | База + добавления → исправленные segments с проверкой M1–M6 |
| `summarize --trust JSON --transcript JSON_OR_TXT [--segments JSON] --out DIR` | Допущенный текст → summary.md + summary.json либо отказ |
| `verify --run DIR` | Повторная проверка артефактов → stdout и 60_verify/report.json |
| `status --run DIR` | Read-only сводка журнала, итоговый код и next action |

`--peer` повторяется. `--out` у run/gate/repair-plan/repair-run/summarize — каталог,
у repair-merge — файл; route принимает файл JSON или каталог. Параметры конкретной
версии уточняйте через `--help`, не конструируйте недокументированные опции.
На фасаде gate используется `--segments`; у отдельного `asr_gate.py` вместо него
`--transcript NAME=PATH` и дополнительно доступен `--evidence NAME=PATH`.

### Отдельные стадии и коды возврата

Общий контракт: `0` ok, `2` partial, `3` failed, `4` blocked/нужен человек,
`64` неверный запуск, `1` операционная/внутренняя ошибка. `run.json.exit_code` —
машинный итог; `verdict` описывает качество текста и не всегда равен исходу выдачи:
например, отказ summary guard переводит итог run в `verdict=blocked`, exit 4,
а провал verify — в `verdict=failed`, exit 3. Публикация требует проверки обоих
полей; verdict отдельного trust при этом может оставаться partial.

Исключения исходного фасада:

- Gate для completed оценки возвращает `0/2`, даже если внутри `verdict=failed`.
- `verify` возвращает `0` при `ok=true`, иначе `3`.
- В исходном `cmd_passthrough` route/summarize разрешены дочерние `0/2`;
  дочерние `3/4` могут превратиться в `E_STAGE_FAILED` и exit 1.
  Если важен исходный `3/4`, вызывайте `.venv/bin/python asr_router.py ...` или
  `.venv/bin/python asr_summary.py ...` напрямую и читайте записанный JSON.
- `repair merge` может вернуть `2` с `MERGE_REJECTED`, сохранив исходную базу;
  отсутствие падения не доказывает улучшение покрытия.
- `--legacy-exit` у run/status объединяет `3/4` в `2`; не используйте в новой интеграции.

## Provisioning и offline-режим

Начальная настройка: `bash scripts/provision.sh` создаёт `.venv` Python 3.12 и
устанавливает `requirements-lock.txt` с `--require-hashes`, только из локального
кэша uv. Нужны заранее установленный uv и локальный Python; подробности и границы
платформы — `README.md`. `bash scripts/provision.sh --online` разрешает сеть
только по явному решению оператора, не устанавливает модели и x20 окружения.
Зависимости pipeline и ML runtime разделены; состав весов — `MODEL_MANIFEST.md`,
решения по правам — `LICENSES.md`.

Обёртка `run.sh` по умолчанию выставляет `HF_HUB_OFFLINE=1` и
`TRANSFORMERS_OFFLINE=1`; `--online` снимает оба флага, но не отменяет
`local_files_only=True` адаптеров. При прямом вызове Python экспортируйте эти
переменные сами. Env-флаги — не запрет сети на уровне ОС.
`ASR_X20_ROOT` — единственный входной корень; `X20_ROOT` синхронизируется как
выходной alias. Прямой x20-адаптер использует тот же env/default из `engines.json`,
не личный датированный путь в коде; если однозначного default нет, требует env.
Для GigaAM поиск соседнего site-packages учитывает major/minor текущего Python.

## JSON-схемы

Версии задаются **точным** строковым полем `schema`. Не принимайте неизвестную
версию за совместимую автоматически. Секунды — числа, `sr` — Гц, SHA-256 —
hex-строка из 64 символов, интервалы — start/end на общей шкале входной записи.
Ниже — минимальные структурные JSON Schema для потребителя (Draft 2020-12).
Они не заменяют семантические проверки, пересчёт сигнала и merge-инварианты.
Разрешены дополнительные диагностические поля для совместимости расширений.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$defs": {
    "interval": {
      "type": "object", "required": ["start", "end"],
      "properties": {"start": {"type": "number", "minimum": 0}, "end": {"type": "number", "minimum": 0}}
    },
    "segments": {
      "title": "asr.segments/1", "type": "object",
      "required": ["schema", "audio", "engine", "segments", "meta"],
      "properties": {
        "schema": {"const": "asr.segments/1"},
        "audio": {
          "type": "object", "required": ["path", "sha256", "duration_s", "sr", "channels"],
          "properties": {
            "path": {"type": "string"}, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "duration_s": {"type": "number", "exclusiveMinimum": 0},
            "sr": {"type": "integer", "minimum": 1}, "channels": {"type": "integer", "minimum": 1}
          }
        },
        "engine": {
          "type": "object", "required": ["name", "model", "version", "params", "level_sensitive"],
          "properties": {
            "name": {"type": "string"}, "model": {"type": ["string", "null"]},
            "version": {"type": ["string", "null"]}, "params": {"type": "object"},
            "level_sensitive": {"type": "boolean"}
          }
        },
        "segments": {
          "type": "array", "minItems": 1,
          "items": {
            "allOf": [{"$ref": "#/$defs/interval"}], "type": "object",
            "required": ["id", "start", "end", "text", "speaker", "conf", "source", "flags"],
            "properties": {
              "id": {"type": "string"}, "text": {"type": "string", "pattern": "\\S"},
              "speaker": {"type": ["string", "null"]}, "conf": {"type": ["number", "null"]},
              "source": {"type": "string"}, "flags": {"type": "array"}
            }
          }
        },
        "meta": {"type": "object"}
      }
    },
    "trust": {
      "title": "asr.trust/2", "type": "object",
      "required": ["schema", "verdict", "primary", "gate_version", "reasons", "audio_metrics", "activity", "language_id", "coverage", "coverage_union", "uncovered_all", "agreement", "config"],
      "properties": {
        "schema": {"const": "asr.trust/2"}, "verdict": {"enum": ["ok", "partial", "failed"]},
        "primary": {"type": "string"}, "gate_version": {"type": "string"},
        "reasons": {
          "type": "array", "items": {
            "type": "object", "required": ["code", "severity", "detail", "intervals"],
            "properties": {
              "code": {"type": "string"}, "severity": {"enum": ["warn", "block"]},
              "detail": {"type": "string"},
              "intervals": {"type": "array", "items": {"$ref": "#/$defs/interval"}}
            }
          }
        },
        "audio_metrics": {"type": "object", "required": ["duration_s", "rms_dbfs", "peak_dbfs", "clip_fraction"]},
        "activity": {"type": "object", "required": ["energy_active_s", "vad_active_s", "noise_floor_dbfs"]},
        "language_id": {"type": "object", "required": ["language", "probability", "method"]},
        "coverage": {"type": "object", "additionalProperties": {"type": "object"}},
        "coverage_union": {
          "type": "object", "required": ["covered_s", "active_s", "coverage_union"],
          "properties": {
            "covered_s": {"type": "number", "minimum": 0}, "active_s": {"type": "number", "minimum": 0},
            "coverage_union": {"type": ["number", "null"], "minimum": 0, "maximum": 1}
          }
        },
        "uncovered_all": {"type": "array", "items": {"$ref": "#/$defs/interval"}},
        "agreement": {"type": ["object", "null"]}, "config": {"type": "object"}
      }
    },
    "route": {
      "title": "asr.route/1", "type": "object",
      "required": ["schema", "verdict", "codes", "blocked", "steps", "budget", "summary_allowed", "summary_mode", "unresolved", "notes"],
      "properties": {
        "schema": {"const": "asr.route/1"}, "verdict": {"enum": ["ok", "partial", "failed", "blocked"]},
        "codes": {"type": "array", "items": {"type": "string"}}, "blocked": {"type": ["object", "null"]},
        "steps": {"type": "array", "items": {"type": "object", "required": ["id", "kind", "because", "params"]}},
        "budget": {
          "type": "object", "required": ["attempts_max", "seconds_max"],
          "properties": {"attempts_max": {"type": "integer", "minimum": 1}, "seconds_max": {"type": "number", "minimum": 0}}
        },
        "summary_allowed": {"type": "boolean"}, "summary_mode": {"enum": ["full", "caveated", "blocked"]},
        "unresolved": {"type": "array"}, "notes": {"type": "array"}
      }
    },
    "repair_plan": {
      "title": "asr.repair.plan/1", "type": "object",
      "required": ["schema", "attempt", "intervals", "total_decode_s", "budget_s", "within_budget", "stop_rule"],
      "properties": {
        "schema": {"const": "asr.repair.plan/1"}, "attempt": {"type": "integer", "minimum": 1},
        "intervals": {
          "type": "array", "items": {
            "allOf": [{"$ref": "#/$defs/interval"}], "type": "object",
            "required": ["id", "start", "end", "pad_s", "reason", "engines", "decode_s"],
            "properties": {"engines": {"type": "array", "minItems": 1, "items": {"type": "string"}}}
          }
        },
        "total_decode_s": {"type": "number", "minimum": 0}, "budget_s": {"type": "number", "minimum": 0},
        "within_budget": {"type": "boolean"}, "stop_rule": {"type": "object", "required": ["min_gain_ratio", "max_attempts"]}
      }
    },
    "summary": {
      "title": "asr.summary/1", "type": "object",
      "required": ["schema", "status", "trusted", "verdict", "codes", "guard", "numbers_sha256", "derived_numbers", "files", "exit"],
      "properties": {
        "schema": {"const": "asr.summary/1"}, "status": {"enum": ["ok", "refused"]},
        "trusted": {"type": "boolean"}, "verdict": {"enum": ["ok", "partial", "failed", "blocked"]},
        "codes": {"type": "array"},
        "guard": {"type": "object", "required": ["numbers_ok", "tasks_quoted", "caveat_present", "phrases_ok", "sections_ok", "violations"]},
        "numbers_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "derived_numbers": {"type": "object"}, "files": {"type": "array", "items": {"type": "string"}},
        "exit": {"enum": [0, 2, 3, 4]}
      }
    },
    "run": {
      "title": "asr.run/1", "type": "object",
      "required": ["schema", "stages", "attempts", "engines", "verdict", "exit_code", "unresolved_s"],
      "properties": {
        "schema": {"const": "asr.run/1"},
        "stages": {
          "type": "array", "items": {
            "type": "object", "required": ["name", "status", "outputs", "exit"],
            "properties": {
              "name": {"type": "string"}, "status": {"enum": ["pending", "done", "skipped", "failed"]},
              "inputs_hash": {"type": "string"}, "outputs": {"type": "array", "items": {"type": "string"}},
              "exit": {"type": "integer"}
            }
          }
        },
        "attempts": {"type": "object"}, "engines": {"type": "object"},
        "verdict": {"enum": [null, "ok", "partial", "failed", "blocked"]},
        "exit_code": {"enum": [null, 0, 1, 2, 3, 4, 64]},
        "unresolved_s": {"type": ["number", "null"], "minimum": 0},
        "effective_transcript": {"type": "string"}, "final_trust": {"type": "string"}
      }
    }
  }
}
```

Для проверки одного документа используйте схему `{"$ref":"#/$defs/segments", ...}`
с тем же `$defs` (аналогично trust/route/repair_plan/summary/run). В пакете нет
новой зависимости jsonschema: используйте валидатор, уже имеющийся у принимающей
стороны. Это контракт, а не заявление о выполненной валидации всех
исторических run-каталогов.

### Семантика и инварианты

- **asr.segments/1:** `start < end`, start монотонны, id уникальны,
  `end <= audio.duration_s + 0.1`, text непустой. `audio.sha256` совпадает
  с manifest. Сырой x20 JSON может иметь пустой SHA: run нормализует его
  в `10_asr/segments.json`; потребляйте нормализованный, не сырой файл.
- **asr.trust/2:** severity обязательна, block → failed. `coverage` хранит
  данные по движкам, но `coverage_union` и `uncovered_all` относятся к **primary**,
  не к общему union всех peers. Покрытие = длина пересечения объединённых
  интервалов primary с энергетической активностью / активность. Нет активности →
  coverage_union=null. VAD off → vad_active_s=null; declared LID не измеряет язык.
  Agreement — token Dice по окнам 20 с primary со **всеми** peers, не WER:
  `per_peer` хранит средний Dice каждого peer, `windows_per_peer` — его окна,
  `mean_dice` — среднее доступных peer-оценок. `LOW_AGREEMENT` определяется
  **худшим** peer при Dice < 0.35, интервалы включают низкие окна всех peers.
  Legacy `pair/windows` относятся к первому peer по алфавиту, не ко всей оценке.
- **asr.route/1:** `blocked` — null либо объект причины; непустой blocked
  запрещает автоматические шаги и summary. `summary_allowed=false` — независимый
  запрет публикации. `budget.seconds_max` по умолчанию 0.35 duration_s.
- **asr.repair.plan/1:** интервалы берутся из `trust.uncovered_all`, дыры ≥1 с,
  padding 0.3 с, объединение близких интервалов, короткие планы отбрасываются;
  выход за бюджет не разрешает декодирование. После merge снова нужен gate.
  M1–M6: overlap ≤0.2 с, вставки в разрешённом интервале ±0.05 с, без повтора
  6-граммы рядом, рост union, не больше 2× базового числа сегментов, монотонность.
- **asr.summary/1:** `status=ok` означает прохождение guard, не trusted=true.
  При partial trusted=false и Markdown содержит `⚠` с кодами. При failed/blocked
  standalone summary создаёт карточку отказа и summary.rejected.md, но не новый
  summary.md. Используйте чистый output: standalone стадия не гарантирует
  удаления summary.md, оставшегося от старого запуска. Full run отдельно
  пропускает/очищает запрещённую summary-стадию.
- **asr.run/1:** state сохраняется между стадиями, поэтому null у итога означает
  незавершённый run. `effective_transcript` и `final_trust` — указатели,
  разрешаемые относительно run-dir; после repair они меняются. outputs в журнале
  относительны этому же каталогу. Не считайте существующий run.json завершением.

## Встраивание в оркестратор

### Запуск подпроцессом

Не собирайте shell-строку из пользовательского ввода, не вызывайте eval и
не импортируйте ML-библиотеки в процесс оркестратора. Изолируйте каталог каждого
job и задавайте общий timeout/ограничения ресурсов отдельно от timeout движка.
Пример на Python stdlib (пути передаёт ваша конфигурация):

```python
import json
import os
from pathlib import Path
import subprocess

root = Path(os.environ["ASR_PY_ROOT"]).resolve()
audio = Path(os.environ["JOB_AUDIO"]).resolve()
out = Path(os.environ["JOB_OUT"]).resolve()  # новый каталог на job
result = subprocess.run(
    ["bash", str(root / "run.sh"), "--audio", str(audio), "--out", str(out),
     "--engine", "whisper_ct2"],
    cwd=str(root), env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    capture_output=True, text=True, timeout=3600, check=False,
)
# stderr — диагностика; stdout может содержать CLI error JSON, это не run.json.
if result.returncode in (1, 64):
    raise RuntimeError(f"ASR operation failed: exit {result.returncode}; inspect protected logs")
state_path = out / "run.json"
if not state_path.is_file():
    raise RuntimeError("ASR did not create run.json")
state = json.loads(state_path.read_text(encoding="utf-8"))
if state.get("schema") != "asr.run/1" or state.get("exit_code") is None:
    raise RuntimeError("Unknown schema or unfinished ASR run")
verification = subprocess.run(
    [str(root / ".venv/bin/python"), str(root / "asr_pipeline.py"), "verify", "--run", str(out)],
    capture_output=True, text=True, timeout=120, check=False,
)
if verification.returncode != 0:
    raise RuntimeError("ASR verify rejected artifacts")
if result.returncode not in (0, 2) or state["exit_code"] not in (0, 2):
    raise RuntimeError("ASR requires human review; do not publish summary")
segments = out / state["effective_transcript"]
trust = out / state["final_trust"]
# summary_allowed, summary_refused, guard and caveat must also be checked before release.
```

TimeoutExpired/отмену обрабатывает ваш job runner; определите остановку всей
группы дочерних процессов, а не только родительского Bash/Python. Не запускайте
два writer в один run-dir. При resume повторно проверьте исходный WAV, конфиг,
указатели и outputs. Наличие/inputs_hash стадий не заменяет проверку содержимого.

### Конфигурация движков без обёртки

Для прямого `asr_pipeline.py run` передайте **уже развёрнутый** JSON-конфиг:
все элементы cmd — реальные argv, переменные `${ASR_PY_ROOT}` и
`${ASR_X20_ROOT:-...}` заменены до вызова; `{audio}`, `{out}`, `{language}`
оставлены ядру. Тот же развёрнутый плоский map нужен `asr_repair.py run`.
Ни загрузчик JSON, ни subprocess сами по себе не дают shell-подстановки env.
Обёртка решает это, сохраняет входной `<out>/.engines.resolved.json` и передаёт
его ядру, которое копирует снимок в `00_input/engines.json`. Не удаляйте
сохраняемый входной конфиг: verify/resume сверяют его fingerprint.
Обёртка также согласует `X20_ROOT` адаптера с `ASR_X20_ROOT` конфига.
При CT2 fallback используйте локальный `whisper` repair-алиас, как run.sh.

Для четырёх гипотез передайте `run --ensemble all` либо список
`--ensemble gigaam,whisper,parakeet,qwen`. Отказ основного ASR, как и любого
участника, допустим и записывается в notes; при двух успешных гипотезах
включается medoid. `run.json.ensemble` содержит engines/votes/mean_confidence/
n_votes/enabled, подробный отчёт — `10_asr/ensemble/report.json`. При одном
успешном движке ансамбль выключается, остаётся его гипотеза независимо от того,
был ли он исходным primary. При нуле — `E_ASR_EMPTY`/операционный отказ.
`ensemble=true` — список участников, не доказательство их выполнения. В текущем
ядре route может предложить normalize/detector/second_engine/arbitrate, но full
run исполняет только repair; остальные шаги фиксируются в `run.json.deferred`.
Принимающая команда должна реализовать их сама или считать неисполненными.

### Откуда брать результат

1. Прочитайте `run.json`, проверьте `schema`, finished state, exit_code и verify.
2. Возьмите текст из `effective_transcript`, оценку — из `final_trust`, маршрут —
   `30_route/route.json`. Не выбирайте файлы по mtime и не используйте старый
   `20_gate/trust.json`, если после ремонта указан другой final_trust.
3. Для выдачи summary требуются непротиворечивые verdict/codes/trusted/guard,
   route.summary_allowed, отсутствие summary_refused и фактический summary.md.
   При `2` публикуйте только с оговоркой и направляйте спорные окна на проверку.
4. При `3/4` оставьте материалы для диагностики, но не выдавайте пересказ как
   допущенный. Не стирайте причину отказа и не превращайте partial в ok.
5. Защитите пути и права на артефакты. Run/json-указатели и конфиги недоверенны
   вне вашего job: валидируйте, что разрешённый путь не выходит из каталога job.

## Проверка перед передачей

Из корня пакета после явного provisioning:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -m pytest -q
bash scripts/smoke.sh
.venv/bin/python scripts/check_portable.py
bash scripts/check_docs.sh
.venv/bin/python asr_pipeline.py verify --run ./run-final
```

Нужны существующий `run-final`, его исходный WAV и сохранённые конфиги. Verify
не запускает ML, но записывает отчёт `60_verify/report.json`; отсутствие входов
означает непроверенный критерий. Для чужой машины нужен свой согласованный run,
не переписывание исторических receipts/fingerprints. `check_docs.sh` выполняет
только проверки документации/файлов и `--help`; provisioning и inference не вызывает.

При partial completed gate/route/summarize и полный run/status возвращают **2**;
verify для корректного partial возвращает **0**, поскольку проверяет инварианты,
а не устраняет оговорки. Gate с verdict failed тоже возвращает 2 — читайте JSON.
Standalone route/summarize могут вернуть 3/4; фасад разрешает дочерние 0/2,
поэтому отказ 3/4 преобразуется в `E_STAGE_FAILED`/1. Отказ guard внутри полного
run даёт blocked/4, провал verify — failed/3. Не считать exit 2 универсальным
разрешением публикации: проверьте итог, guard, caveat и route.summary_allowed.

## Чеклист приёмки

Это **критерии передачи**, а не заранее поставленные отметки «принято».

1. [ ] На целевой машине есть Python 3.12/.venv и разрешённые локальные веса; никто не копирует старую .venv между машинами, не устанавливает и не скачивает пакеты неявно.
2. [ ] Provisioning выполнен явно через `scripts/provision.sh`; полный hash-locked transitive набор CPython 3.12/macOS arm64 соответствует `requirements-lock.txt`, внешние x20 runtime сверены отдельно с `MODEL_MANIFEST.md`.
3. [ ] `bash scripts/smoke.sh` проходит без ML, сети и внешнего WAV: gate/route/summary возвращают ожидаемый exit 2, артефакты и caveat проверены.
4. [ ] `.venv/bin/python scripts/check_portable.py` даёт 0 ошибок; все Python-пути переносимы, интерпретаторы/файлы cmd существуют. Для CT2-only отдельно проверен `--engine whisper_ct2` и не заявлена готовность отсутствующих MLX runtime.
5. [ ] Потребитель различает версии всех шести JSON-схем, незавершённый run, exit 0/2/3/4/64/1 и читает effective_transcript/final_trust вместо поиска по имени/mtime.
6. [ ] Gate считает union, а не сумму перекрывающихся сегментов; дыры, declared/off и независимый LID не смешаны; покрытия/согласия не названы WER или точностью.
7. [ ] Route блокирует LANGUAGE_MISMATCH/UNKNOWN_REASON, сохраняет summary_allowed; неисполненные deferred-шаги не выданы за выполненные.
8. [ ] Repair соблюдает бюджет и M1–M6, затем выполняется повторный gate; rejected merge не теряет базу. Resume проверен с изменёнными/удалёнными артефактами, а не только с неизменным входом.
9. [ ] **summary не создаётся при failed**: проверены чистый и повторный run, отсутствие устаревшего summary.md, отказ guard, caveat при partial и trusted=false вне ok.
10. [ ] **verify пересчитывает покрытие** из исходного сигнала/интервалов, а не доверяет coverage_union/uncovered_all: согласованно подменённые trust+run и неверное покрытие выявляются; отдельно проходит `.venv/bin/python -m pytest -q`.

### Границы приёмки и известные пробелы

Текущий `verify_run` независимо читает WAV, пересчитывает энергетическую
активность и покрытие из соответствующих первичных/исправленных segments,
сверяет coverage_union, primary coverage и uncovered_all. Расхождения дают
`E_COVERAGE_MISMATCH`; дополнительно проверяются fingerprints стадий, schemas,
порядок и summary. Принимающая команда должна повторить отрицательные сценарии
пункта 10 на своём candidate: наличие согласованных JSON не является проверкой.

Portability check запрещает персональные пути в Python, включая tests, но не
синтетические `/x`, `/abs` и переносимые системные пути. Единственное разрешённое
конфигурационное исключение — default `ASR_X20_ROOT` в `engines.json`;
адаптер не дублирует его в Python. На новой машине задайте собственный корень.
Живые модели/целевое оборудование и юридические условия отдельно не проверены.
`MODEL_MANIFEST.md` фиксирует локальные размеры/хеши/revisions и пакеты x20;
`LICENSES.md` оставляет исходники и Whisper MLX открытым human gate.
