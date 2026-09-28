# apps/

## `emotionedge` (CLI)

```bash
emotionedge demo                      # blueprint walkthrough: synthetic angry "I can't believe you did this!" -> Hindi
emotionedge demo --conversation       # anger, sadness, joy
emotionedge demo --target es --realtime
emotionedge run --input speech.wav --script words.json   # stand-in ASR fed by a word script
emotionedge run --input speech.wav --config config/pipeline.engines.yaml   # real engines
emotionedge live                      # microphone -> speaker (build with -DEE_WITH_MINIAUDIO=ON)
emotionedge say --engine piper --model-id tts.piper.en_US.lessac.medium --text "One. | Two." --out speech.wav
                                      # test input for real ASR: '|' separates utterances (--gap seconds)
emotionedge models verify             # SHA-256 check of models/manifest.json
emotionedge stages                    # registered stage types
```

Outputs go to `--out DIR` (default `out/<command>`):

- `translated.wav`: the translated speech.
- `mix.wav`: written with `--mix-source-db`. The original speech sits under the translation and is
  ducked while the translation plays.
- `captions.srt`: live captions with emotion tags.
- `session.json`: per-utterance transcript, emotion, MT input, translation, prosody plan, ECS and
  latency budget.
- `metrics.prom`: Prometheus text exposition.
- `trace.json`: written with `--trace`. Open it in ui.perfetto.dev.

Any setting can be overridden with `--set stage.param=value`, for example
`--set translate.engine=ct2`. `say` takes `--set tts.param=value`.

Devices:
- `--set pipeline.device=cpu` (or `cuda`, or `auto`, which the engines config uses) applies to
  every stage that does not name its own `device`.
  - `auto` uses CUDA when the build has it, and otherwise stays on the CPU without a warning.
  - `cuda`, or `auto` in a CUDA build, warns when it has to fall back to the CPU: no GPU, or
    CUDA/cuDNN libraries missing from the loader path.
- `--set emotion.acoustic_device=cuda` / `lexical_device` place a stage's two models separately.
- `--set tts.ort_profile=out/kokoro` writes ONNX Runtime's per-node profile (op, device, time),
  e.g. to find nodes that leave the GPU.
- `gpu_id` picks the GPU.

## Planned (roadmap phase 5 "Ship")

- **Desktop UI** (Dear ImGui): live captions with emotion tags, device selection, latency panel.
- **Server**: gRPC / WebSocket streaming API over the same `Session`/`Graph` runtime.
