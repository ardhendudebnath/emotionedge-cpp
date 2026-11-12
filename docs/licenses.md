# Licences: what each build may be used for

This page lists everything a build of EmotionEdge links or loads, with the licence each upstream source states. All of them were checked against their sources on 2026-11-12. It reads those licences as written. It is not legal advice, so check again before shipping anything.

## In short

| Build | What it adds | What its licences allow |
|---|---|---|
| Stand-in engines (the default CMake options) | yaml-cpp and nlohmann/json; no models | Any use. Everything in it is MIT, BSD, Apache-2.0 or zlib. |
| en→hi, GPU (`config/pipeline.engines.yaml`, language pack `en-hi`) | the real engines and the `en-hi` models | **Non-commercial** use only, under three separate constraints (below) |
| en→hi, fast CPU profile (`config/pipeline.cpu-fast.yaml`) | the same, with Piper's `pratham` voice instead of Kokoro | Non-commercial use only. The same three constraints apply, and the voice's training data is non-commercial too. |
| hi→en (language pack `hi-en`) | the real engines, plain NLLB and Piper's `lessac` voice | **Research only**: on top of the en→hi constraints that apply, the voice's training data is licensed for research alone |

The three constraints on the en→hi builds:
1. **NLLB-200 is CC-BY-NC-4.0.** Every build that translates uses it, and the emotion-token LoRA is derived from it.
2. **The DistilRoBERTa emotion model states no licence.** Without one, nobody has been granted the right to redistribute it, so shipping it needs its author's permission or a replacement model.
3. **espeak-ng is GPL-3.0-or-later.** `EE_WITH_PIPER` links it, for Piper's phonemes and for Kokoro's Hindi G2P. A binary built with it can be distributed only under the GPL-3.0's terms.

**This repository has no licence file of its own.** Its code is therefore all rights reserved: anyone can read it on GitHub, but nobody has been given permission to copy, change or share it. Choosing a licence is the owner's call. espeak-ng limits how binaries are distributed, not which licence the source can have: MIT or Apache-2.0 source can still be built into a GPL-3.0 binary.

## Models

Every model the registry (`models/manifest.json`) can load:

| Model id | Used by | Licence | Training data, as its card states |
|---|---|---|---|
| `vad.silero.v6` | every pack | MIT | — |
| `asr.whisper.base.q5_1` | every pack | MIT (whisper.cpp's conversion of OpenAI Whisper) | — |
| `mt.nllb200.distilled600m.int8` | `hi-en`; `en-hi` without the LoRA | **CC-BY-NC-4.0** | — |
| `mt.nllb200.distilled600m.emo.int8` | `en-hi` | **CC-BY-NC-4.0** (derived from NLLB-200) | Our MT corpus (see "Data") |
| `tts.kokoro.82m.hi` | `en-hi` | Apache-2.0 | Permissive or public-domain audio. Two of its sets are CC BY 3.0 (Koniwa) and CC BY 4.0 (SIWIS), which require attribution; its card names them. |
| `tts.piper.hi_IN.pratham.medium` | fast CPU profile | The voice repository is MIT | **CC BY-NC-SA 4.0**, per the voice's MODEL_CARD |
| `tts.piper.en_US.lessac.medium` | `hi-en`; the English input of `ml/eval/benchmark.py` | The voice repository is MIT | **Blizzard 2013 Lessac data: research only, no commercial use, no redistribution** |
| `speaker.ecapa.voxceleb` | `en-hi` | Apache-2.0 | VoxCeleb |
| `emotion.acoustic.emotion2vec_plus_base` | every pack | **FunASR Model License 1.1** (see the note below) | — |
| `emotion.lexical.distilroberta_en` | `en-hi` | **None stated.** Its card asks users to cite Hartmann (2022). | Six datasets, among them MELD (GPL-3.0), GoEmotions and ISEAR |

**The FunASR Model License (emotion2vec+)**:
- It grants the right to use, copy, modify and share the model.
- It requires keeping the source, the author and the model names.
- It neither grants nor excludes commercial use. Its disclaimer calls the software one provided for reference and learning.
- The licence ends automatically on any breach, including its clause against disparaging the software.
- Alibaba may revise the terms, and continued use counts as accepting them.

**Files learned here and shipped with Kokoro** (the style offsets and prosody plans in its voice directory) were fitted to Kokoro's speech of the MT corpus's Hindi, judged by emotion2vec+.

## Libraries

By build option:

| Option | Library (version) | Licence | Notes |
|---|---|---|---|
| always | yaml-cpp (0.9.0), nlohmann/json (3.12.0) | MIT | fetched |
| tests only | GoogleTest | BSD-3-Clause | not part of any app |
| `EE_WITH_ONNXRUNTIME` | ONNX Runtime | MIT | a release package |
| `EE_WITH_WHISPER` | whisper.cpp and ggml (v1.9.4) | MIT | fetched |
| `EE_WITH_CTRANSLATE2` | CTranslate2 (4.8.2) | MIT | Installed separately. CI builds it with oneDNN (Apache-2.0) and OpenBLAS (BSD-3-Clause); a build with Intel MKL adds Intel's licence. |
| | SentencePiece (0.2.0, or the system's) | Apache-2.0 | |
| `EE_WITH_PIPER` | **espeak-ng** | **GPL-3.0-or-later** | Piper's phonemes and Kokoro's Hindi G2P |
| | pcaudio (espeak-ng's audio output) | GPL-3.0-or-later | comes with espeak-ng |
| | sonic (espeak-ng's speed control) | Apache-2.0 | comes with espeak-ng |
| GPU builds | CUDA, cuBLAS, cuDNN | NVIDIA's licence agreements | installed by the user; not part of this repository |
| `EE_WITH_MINIAUDIO` | miniaudio (0.11.25) | public domain (Unlicense) or MIT-0 | It loads the system's audio libraries when it runs, e.g. PulseAudio (LGPL-2.1+). |
| `EE_WITH_WEBSOCKET` | IXWebSocket (12.0.1) | BSD-3-Clause | fetched |
| | OpenSSL 3 (for TLS) | Apache-2.0 | the system's |
| `EE_WITH_GRPC` | gRPC | Apache-2.0 | the system's, with its dependencies: protobuf (BSD-3-Clause), abseil (Apache-2.0), re2 (BSD-3-Clause) and c-ares (MIT) |
| `EE_WITH_DESKTOP` | Dear ImGui (1.92.9b) | MIT | fetched |
| | GLFW (3.5.1) | zlib | fetched |
| | HarfBuzz (14.5.1) | MIT ("Old MIT") | fetched |
| | Noto Sans Devanagari | SIL OFL 1.1 | The build copies `OFL.txt` next to the font. |

Distributing binaries brings two duties:
- The permissive licences (MIT, BSD, zlib, Apache-2.0) require their notices to go with the binary.
- The GPL-3.0 requires offering the corresponding source.

## Data

Datasets used to train or tune something that ships, or to measure it:

| Dataset | Licence | Used for | Part of what ships |
|---|---|---|---|
| MELD | GPL-3.0 | The English source of the MT corpus (train and dev); its test split evaluates the lexical model | The NLLB LoRA was trained on it |
| GoEmotions | Apache-2.0 | The English source of the MT corpus | The NLLB LoRA was trained on it |
| NLLB-200 1.3B translations | Output of a CC-BY-NC-4.0 model | The Hindi targets of the MT corpus, and the sentences the Kokoro style offsets and plans were fitted on | The LoRA, the offsets and the plans |
| RAVDESS | CC BY-NC-SA 4.0 (commercial licences are sold separately) | Evaluation, and calibrating the barge-in thresholds and the judges | Only numbers |
| FLORES-200 | — | MT evaluation (chrF/BLEU) | Nothing |

## Tools that don't ship

`ml/` runs offline and nothing in it is loaded by the runtime. Two of its models carry their own limits:
- **audeering's V·A·D judge** is CC BY-NC-SA 4.0.
- **The Llama 3.1 8B used for the LLM rewrites** is under Meta's Llama 3.1 Community License.
