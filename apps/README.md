# apps/

## `emotionedge` (CLI)

```bash
emotionedge demo                      # blueprint walkthrough: synthetic angry "I can't believe you did this!" -> Hindi
emotionedge demo --conversation       # anger, sadness, joy
emotionedge demo --target es --realtime
emotionedge run --input speech.wav --script words.json   # stand-in ASR fed by a word script
emotionedge run --input speech.wav --config config/pipeline.engines.yaml   # real engines
emotionedge live                      # microphone -> speaker (build with -DEE_WITH_MINIAUDIO=ON)
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
`--set translate.engine=ct2`.

## Planned (roadmap phase 5 "Ship")

- **Desktop UI** (Dear ImGui): live captions with emotion tags, device selection, latency panel.
- **Server**: gRPC / WebSocket streaming API over the same `Session`/`Graph` runtime.
