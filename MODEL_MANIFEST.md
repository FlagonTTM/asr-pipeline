# Манифест моделей и внешних runtime

## Область и способ проверки

Пути ниже относительны `${ASR_X20_ROOT}/work/models`. Модели и x20 venv не входят
в исходники и не устанавливаются `scripts/provision.sh`. Корень передаёт оператор
через `ASR_X20_ROOT`; датированный конфигурационный fallback `engines.json`
сохранён только для совместимости текущей рабочей машины, не для новой установки.

Источники — локальные `receipt.json` и model cards. Сеть и ML не запускались.
Размеры — байты, SHA-256 пересчитаны по фактическим файлам. Для Whisper/Parakeet
сверены все `files[].bytes/sha256`; для GigaAM/Qwen — размеры `siblings[].size`
и SHA-256 LFS-весов. Для прочих файлов этих двух моделей SHA-256 вычислен сейчас:
`blobId` в receipt — Git object ID, его нельзя выдавать за SHA-256 файла.
HF revision взята из `revision` либо `sha` receipt; это локальная provenance,
не независимое подтверждение полного upstream snapshot.

## Сводка

| Каталог | HF repo | Revision | Файлы без receipt, байт |
|---|---|---|---:|
| `gigaam_hf` | `ai-sage/GigaAM-v3` | `c7f128b8accdd9624df905e5c2d7b7a48c27c0d8` | 446438756 |
| `whisper` | `mlx-community/whisper-large-v3-turbo` | `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` | 1613978239 |
| `parakeet` | `mlx-community/parakeet-tdt-0.6b-v3` | `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` | 2509042622 |
| `qwen3_asr_4bit` | `mlx-community/Qwen3-ASR-1.7B-4bit` | `78a389c776a5483b2d0d4ea5494e11012e0d6159` | 1607631587 |

## gigaam_hf

Путь: `${ASR_X20_ROOT}/work/models/gigaam_hf`.

`receipt.json`: 1757 байт; SHA-256 `3642063056f770a5b52116983041ac5388f039ac1152cd79737596d97e30a38e`.

| Файл | Размер, байт | SHA-256 | Сверка с receipt |
|---|---:|---|---|
| `README.md` | 24 | `d8d7a46d41a1a37fe4f0a5f637bf55c649310185329127d8a2204632e480be17` | размер совпал; SHA-256 вычислен локально |
| `config.json` | 2358 | `ca2998f00f064648988111c426deaa4c8bc86872a8bbe443598f2b1511dc0947` | размер совпал; SHA-256 вычислен локально |
| `modeling_gigaam.py` | 49135 | `269be43b635b1e510115baa2a843c5cbaa052e8adf0be30dc133a2ba5b5f2d86` | размер совпал; SHA-256 вычислен локально |
| `pytorch_model.bin` | 446387239 | `a89b662052ad2b8129b334ab45c171b04fc6699d69bd6917575013ff79172ec2` | размер и SHA-256 совпали |

`.gitattributes` указан upstream receipt, но локально отсутствует; он не нужен
адаптеру и не включён в сумму. Это не полный зеркальный snapshot репозитория.

## whisper

Путь: `${ASR_X20_ROOT}/work/models/whisper`.

`receipt.json`: 575 байт; SHA-256 `ed63246722bbba288da5149381e2461d6f31529a4a40dadb2918b71d7626baa8`.

| Файл | Размер, байт | SHA-256 | Сверка с receipt |
|---|---:|---|---|
| `weights.safetensors` | 1613977612 | `951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6` | размер и SHA-256 совпали |
| `README.md` | 359 | `78f2d3dc65072211d957fc1b646c2d96e8e8f96dd45ceb2ac5f2643d0c5c05da` | размер и SHA-256 совпали |
| `config.json` | 268 | `b34fc29e4e11e0a25e812775dd67f4dd16fc2c8eb43d28ae25ff7d660ecb6379` | размер и SHA-256 совпали |

## parakeet

Путь: `${ASR_X20_ROOT}/work/models/parakeet`.

`receipt.json`: 1024 байт; SHA-256 `63fbfd775b341b6a1270ca870f99994b0d1c7b7b448835cdb5a88e8a068003b6`.

| Файл | Размер, байт | SHA-256 | Сверка с receipt |
|---|---:|---|---|
| `model.safetensors` | 2508288736 | `05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592` | размер и SHA-256 совпали |
| `README.md` | 1081 | `724e3ba46903986ff6caeb8fddb7495c06b5a7a8d908f11e43cc0a7e136b5e23` | размер и SHA-256 совпали |
| `config.json` | 244093 | `f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c` | размер и SHA-256 совпали |
| `tokenizer.model` | 360916 | `eacec2b0a77f336d4a2ca4a25a7047575d3c2b74de47e997f4c205126ed3135e` | размер и SHA-256 совпали |
| `tokenizer.vocab` | 101024 | `41130ff456706304a1adec782ccc9e003c4d417e8e324353d281be958cac4e17` | размер и SHA-256 совпали |
| `vocab.txt` | 46772 | `3cde1409fd78783a79b29ed4d32da57c746993856f7c8263bcb905d2e5839db7` | размер и SHA-256 совпали |

## qwen3_asr_4bit

Путь: `${ASR_X20_ROOT}/work/models/qwen3_asr_4bit`.

`receipt.json`: 4228 байт; SHA-256 `a6cd6b4ae3ef62942a6d09cd5044b9172062ff275dba140ae051b639cbbffafc`.

| Файл | Размер, байт | SHA-256 | Сверка с receipt |
|---|---:|---|---|
| `README.md` | 1008 | `4d7f16d0a01cd626a3d5291b77249f135eeedd9b1b259d17716cc6b3ef7365cb` | размер совпал; SHA-256 вычислен локально |
| `chat_template.json` | 1161 | `75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff` | размер совпал; SHA-256 вычислен локально |
| `config.json` | 7188 | `539c6c9d482349066e2e740241e39896f3bf4de0866c245f6539c85dbb19d93c` | размер совпал; SHA-256 вычислен локально |
| `generation_config.json` | 142 | `1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c` | размер совпал; SHA-256 вычислен локально |
| `merges.txt` | 1671853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` | размер совпал; SHA-256 вычислен локально |
| `model.safetensors` | 1603081617 | `9848eaf7a5c1589c671b35035ac27b72e248dd0c604eacae547e7e403d29db45` | размер и SHA-256 совпали |
| `model.safetensors.index.json` | 78968 | `2612ab715223843b3dec0742737bc85f50914ca1484ade45955f99b4e012f2bb` | размер совпал; SHA-256 вычислен локально |
| `preprocessor_config.json` | 330 | `45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d` | размер совпал; SHA-256 вычислен локально |
| `tokenizer_config.json` | 12487 | `4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c` | размер совпал; SHA-256 вычислен локально |
| `vocab.json` | 2776833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` | размер совпал; SHA-256 вычислен локально |

`.gitattributes` указан upstream receipt, но локально отсутствует; он не нужен
адаптеру и не включён в сумму. Это не полный зеркальный snapshot репозитория.

## Требования x20 venv

Apple Silicon/macOS, совместимый MPS/Metal; Python определяется соответствующим
venv. `.venv` pipeline — отдельная среда CPython 3.12. Списки ниже не являются
install-lock: нет хешей дистрибутивов и доказанного provisioning x20. За
воспроизведение этих окружений отвечает поставщик x20. Не копируйте готовую venv:
восстановите её на целевой машине и отдельно проверьте inference с разрешённым аудио.

GigaAM: `${ASR_X20_ROOT}/work/ru_asr/venv/bin/python`; нужны `torch`, `torchaudio`,
`numpy`, `soundfile`, `sentencepiece`, `omegaconf`, `hydra-core`. `transformers`
и зависимости берутся из соседнего `work/venv`, найденного через
`lib/python*/site-packages` с major/minor текущего Python; адаптер не фиксирует
имя `python3.12` и не запускает чужой interpreter.

Whisper/Parakeet/Qwen: `${ASR_X20_ROOT}/work/venv/bin/python`; нужны `mlx`,
`mlx-metal`, `mlx-whisper`, `mlx-audio`, `transformers`, `tokenizers`, `safetensors`,
`huggingface-hub`, `numpy`, `soundfile`, `sentencepiece`, `tiktoken`. Этот venv
содержит также `torch` для зависимостей runtime.

Ниже полный локально прочитанный список distribution metadata каждого venv
(без импорта ML). Это наблюдаемые версии, не минимальные version ranges.

### work/ru_asr/venv

```text
antlr4-python3-runtime==4.9.3
cffi==2.1.1
coloredlogs==15.0.1
filelock==4.2.9
flatbuffers==25.12.19
fsspec==2026.9.0
gigaam==0.2.0
humanfriendly==10.0
hydra-core==1.3.8
Jinja2==3.1.6
MarkupSafe==3.0.4
ml_dtypes==0.6.0
mpmath==1.3.0
networkx==3.7
numpy==2.5.4
omegaconf==2.3.1
onnx==1.19.1
onnxruntime==1.23.2
packaging==26.3
protobuf==7.36.2
pycparser==3.12
PyYAML==6.0.3
sentencepiece==0.2.3
setuptools==81.0.0
soundfile==0.14.0
sympy==1.14.0
torch==2.11.0
torchaudio==2.11.0
tqdm==4.70.1
typing_extensions==4.16.0
```

### work/venv

```text
annotated-doc==0.0.5
anyio==4.15.1
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.0
filelock==4.2.4
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.7.0
httpcore2==2.16.0
httpx2==2.16.0
huggingface_hub==2.3.0
idna==3.20
Jinja2==3.1.6
llvmlite==0.50.0
markdown-it-py==4.2.0
MarkupSafe==3.0.4
mdurl==0.1.2
miniaudio==1.71
mlx==0.32.4
mlx-audio==0.6.2
mlx-metal==0.32.4
mlx-whisper==0.4.3
more-itertools==11.2.1
mpmath==1.3.0
networkx==3.7
numba==0.68.0
numpy==2.5.4
packaging==26.3
pip==25.3
pycparser==3.12
Pygments==2.21.0
PyYAML==6.0.3
RapidFuzz==3.14.6
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scipy==1.18.1
sentencepiece==0.2.2
setuptools==84.2.1
shellingham==1.5.4
sounddevice==0.5.6
soundfile==0.14.0
sympy==1.14.0
tiktoken==0.14.0
tokenizers==0.23.3
torch==2.14.1
tqdm==4.70.1
transformers==5.21.0
truststore==0.10.5
typer==0.27.3
typing_extensions==4.16.0
urllib3==2.8.0
```

## Не включено

CT2 `large-v3`, веса LID/VAD и необязательные модели diarization/enhancement
не идентифицированы этим манифестом как готовые к запуску. Для включения нужны
отдельные repo+revision, локальные пути, размеры, SHA-256 и решение по лицензиям.
Перед выдачей сверьте этот манифест и `LICENSES.md` с фактической поставкой.
