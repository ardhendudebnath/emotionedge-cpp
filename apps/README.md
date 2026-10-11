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
emotionedge serve --port 8080 --grpc-port 50051   # WebSocket and gRPC, one live session per client
emotionedge stream --input speech.wav             # gRPC client for serve (-DEE_WITH_GRPC=ON)
emotionedge-desktop                   # desktop app: live captions, emotion, latency (-DEE_WITH_DESKTOP=ON)
emotionedge run --input speech.wav --config config/pipeline.engines.yaml --realtime --echo-sim -6 \
    --set frontend.record=out/echo/run   # what an open loudspeaker would do, recorded
```

Echo (front end, 1.2):
- `--echo-sim DB` (with `--realtime`) feeds what is played back into the input through a
  simulated loudspeaker and room: soft clipping, `--echo-rt60-ms` of reverberation (250) and
  `--echo-delay-ms` of delay (40).
- `--set frontend.record=PREFIX` writes what the pipeline hears (`PREFIX_mic.wav`, 16 kHz). When
  audio plays, it also writes what played, aligned sample by sample with the microphone
  (`PREFIX_ref.wav`): the input an echo canceller works from.

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

## `emotionedge serve` (streaming APIs, 5.3)

Two transports share one pool of live sessions, so the warm sessions and the session limit are
for both:

| Transport | Build | Option | Clients |
|---|---|---|---|
| WebSocket | `-DEE_WITH_WEBSOCKET=ON`: fetches IXWebSocket v12.0.1 (BSD-3), pinned; TLS with OpenSSL if found, no zlib | `--port 8080` (0: off) | browsers, anything with WebSockets |
| gRPC | `-DEE_WITH_GRPC=ON`: an installed gRPC and protobuf (apt: `libgrpc++-dev protobuf-compiler-grpc libprotobuf-dev protobuf-compiler`) | `--grpc-port 50051` (default: off) | generated from [`proto/emotionedge/v1/translator.proto`](../proto/emotionedge/v1/translator.proto), any language |

```bash
emotionedge serve --config config/pipeline.engines.yaml --port 8080 --grpc-port 50051 [--host 127.0.0.1] [--max-sessions 1] [--warm 1]
python apps/web/stream_wav.py speech.wav --url ws://127.0.0.1:8080/   # WebSocket reference client (pip install websockets)
emotionedge stream --input speech.wav --grpc 127.0.0.1:50051         # gRPC reference client (C++, typed stubs)
```

Both listen on localhost, unencrypted and open, by default. Before exposing them, add TLS and a
token (see below).

Open `apps/web/index.html` for a browser client. It streams the microphone or a WAV file, and
shows live captions with the emotion and ECS while the translation plays. Serve the folder
(`python -m http.server -d apps/web`) if the browser refuses the microphone on `file://`.

Each client gets one live session, the same threaded graph as `live` (`core/server/session_pool.cpp`):
- **The speaker is the client.** The translated speech goes out at playback pace, as a sound card
  would take it, so adaptive pacing and barge-in behave as they do on a device.
- **Sessions are kept warm.** Loading a session's models takes seconds, so the server keeps
  `--warm` sessions (default 1) loaded ahead of time. A connection takes one and gets `ready` at
  once. The replacement loads when a session ends, not while one runs.

  On the real engines (jfk.wav, 6 connections per setting, interleaved):

  | Setting | `ready` after connecting | End-to-end p95 |
  |---|---|---|
  | `--warm 0` | 2.9–6.1 s | 690–729 ms |
  | `--warm 1` | 0.03–0.06 s | 628–858 ms |

  The warm medians match the cold ones, with a wider spread. Loading the replacement during the
  next session instead added 300–450 ms to the p95 of half the connections.
- **Sessions are never reused.** A session learns its speaker: the voice print, the emotion
  state, the closed-loop correction. When the connection ends, its session is discarded.

  Its memory then goes back to the system (`malloc_trim`, on glibc). Its threads free the
  engines' buffers into glibc's per-thread arenas, which keep the pages otherwise. Over 10
  sessions, one at a time, RSS went from 1.8 GB to a 3.0–3.6 GB plateau without the trim, and
  to 2.1–2.3 GB with it. After 8 sessions with 3 at once: 3.9–4.1 GB without, 2.8 GB with.
- **Models are shared.** Each model is loaded once per process, and every session uses it:
  - **Whisper:** one context per model and device, with a decoder state per session.
  - **NLLB:** one CTranslate2 translator, with a replica per session that may run at once
    (`--max-sessions`). Replicas share the weights, so sessions translate in parallel. Set
    `translate.replicas` to override.
  - **ONNX Runtime models:** one session per model.

  A session still holds its own state: its voice print, emotion state and decoder buffers.
  `--max-sessions` refuses connections beyond the limit; warm sessions come on top of it.
  `--warm 0` loads on connect instead.

  On the real engines (RTX 5070 Ti, jfk.wav, `--max-sessions 3`), against each session loading
  its own models:

  | | Own models | Shared |
  |---|---|---|
  | GPU memory, 3 sessions loaded | 6.6 GB | 5.1 GB |
  | Each session after the first | 1.9 GB | 0.6 GB (the NLLB replicas are allocated with the first) |
  | 1 client, end to end p50 | 389–438 ms | 397–414 ms |
  | 3 clients at once, p50 | 1130–1360 ms | 1163–1393 ms |

  Transcripts and translations are the same either way.
- **Several clients at once are slower.** Three at once on one GPU take about 3× as long per
  utterance. Most of it is the final translation pass (~50 ms alone, 400–600 ms with three).
  It waits on the GPU behind the other sessions' work.

  Skipping the wait-k drafts while other sessions translate made this worse. Three at once went
  from 4.3 to 6.4 s p50, and the final pass from 1.5 to 3.2 s, interleaved over 3 blocks on a
  slow day for this laptop. The drafts translate most of the sentence ahead of time. The final
  pass keeps their prefix, which CTranslate2 reads in one step, and decodes only the rest word
  by word. Without drafts, the whole sentence is decoded word by word on the busy GPU.

The WebSocket protocol, on `ws://HOST:PORT/?rate=16000`, where `rate` is the client's sample rate
(sessions run at 16 kHz; the server resamples other rates):

| Direction | Frame | Content |
|---|---|---|
| client → server | binary | mono 16-bit little-endian PCM at `rate` |
| client → server | text | `{"type": "end"}`: no more speech; finish, play out, then `done` |
| server → client | text | `{"type": "ready", input_rate, output_rate, source_language, target_language}` once the models are loaded. Audio sent earlier is buffered, up to 30 s |
| server → client | text | one event per result, each with `type` and `utterance`: `transcript` (partial and final), `emotion` (V·A·D, label, confidence, emphasis), `translation` (drafts and the final), `prosody` (the controller's plan), `consistency` (ECS and what 5.2 heard, per clause), `playout` |
| server → client | text | `{"type": "done", "session": {...}}`: the session export, after the last audio. Or `{"type": "error", message}` |
| server → client | binary | the translated speech: mono 16-bit little-endian PCM at `output_rate` (24 kHz), in 10 ms blocks. Silence is not sent |

Measured through the server on the real engines (RTX 5070 Ti), the server's own end-to-end
telemetry matches in-process runs ([docs/benchmark.md](../docs/benchmark.md)):

| Input | End to end through the server | In-process benchmark |
|---|---|---|
| jfk.wav | p50 373 ms, p95 610 ms | p50 381–389 ms, p95 628–647 ms |
| 3 back-to-back sentences | p50 877 ms, p95 1655 ms | p50 844–893 ms, p95 1648–1687 ms |

The client saw 360–442 ms from the end of an utterance to its first translated audio, where no
earlier translation was still playing.

### gRPC

The `emotionedge.v1.Translator` service has one bidirectional streaming call, `Translate`, with
the same content as the WebSocket protocol, typed:

| | Messages |
|---|---|
| requests | an optional `StreamConfig {sample_rate}`, then `audio` (16-bit PCM bytes), then `EndOfSpeech` |
| responses | `Ready`, then as they happen `Transcript`, `Emotion` (an `EmotionPoint`: label, V·A·D), `Translation`, `Prosody`, `Consistency` (ECS and the `EmotionPoint` heard), `Playout`, and `audio` (the translated speech, 16-bit PCM at `Ready.output_rate`); last `Done {session_json}` or `Error` |
| status | `RESOURCE_EXHAUSTED` beyond `--max-sessions`, `INVALID_ARGUMENT` for a sample rate outside 8–192 kHz, `CANCELLED` when the client hangs up |

Generate a client for any language from the proto, for example
`python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. emotionedge/v1/translator.proto`.
`emotionedge stream` (`apps/cli/main.cpp`) is a C++ one: it streams a WAV at speaking pace and
prints the captions.

On the real engines (jfk.wav, warm session):

| Transport | End to end p50 | End to end p95 |
|---|---|---|
| gRPC, 2 calls | 381–389 ms | 595–631 ms |
| WebSocket, same server | 471 ms | 721 ms |

Both are in line with the in-process benchmark.

**Building gRPC with the engines:** Ubuntu's SentencePiece (for the NLLB tokenizer) carries its
own copy of protobuf and exports it. Next to gRPC's protobuf, that crashed the process. So with
`EE_WITH_GRPC` and `EE_WITH_CTRANSLATE2`, the build fetches SentencePiece v0.2.0 (pinned) and
builds it against the system protobuf, with its own abseil kept private.

### TLS and a token

Both transports take the same settings:

```bash
emotionedge serve --host 0.0.0.0 --port 8443 --grpc-port 50051 \
    --tls-cert server.pem --tls-key server.key --token-file token.txt
emotionedge stream --input speech.wav --grpc myhost:50051 --tls-ca ca.pem --token "$(cat token.txt)"
python apps/web/stream_wav.py speech.wav --url wss://myhost:8443/ --ca ca.pem --token "$(cat token.txt)"
```

| Setting | WebSocket | gRPC |
|---|---|---|
| `--tls-cert` / `--tls-key` (PEM) | `wss://` (IXWebSocket with OpenSSL; a build without OpenSSL refuses to start with them) | SSL server credentials |
| `--token` / `--token-file` | `Authorization: Bearer <token>` on the handshake, or `?token=` from browsers, which cannot set the header (the browser client has a token field). Refused: an `unauthorized` error, then close code 1008 | `authorization: Bearer <token>` metadata; refused with `UNAUTHENTICATED` |

- **No session is taken** for a refused client.
- **Tokens are compared in constant time.** `--token-file` keeps the token out of the process
  list.
- **Prefer the header to `?token=`:** a URL can end up in proxy logs and browser history.
- **A warning is printed** when `--host` is reachable from other machines but TLS or the token
  is missing. The token is only worth something over TLS.
- **No latency cost.** On the real engines, a plain and a secure server ran side by side (jfk.wav,
  5 interleaved runs each, medians):

  | | `ready` after connecting | End-to-end p50 / p95 |
  |---|---|---|
  | gRPC, plaintext | 0.00 s | 389 / 611 ms |
  | gRPC, TLS + token | 0.01 s | 373 / 614 ms |
  | WebSocket, `ws://` | 0.03 s | 397 / 614 ms |
  | WebSocket, `wss://` + token | 0.05 s | 389 / 631 ms |
- **Test certificates:** `tests/data/tls` holds a test-only CA and a localhost certificate for
  the tests. Its key is public: never serve with it.

Limit: **a client that stops reading** stalls its own session's playback.

## `emotionedge-desktop` (desktop app, 5.3)

![The desktop app after three sentences, real engines on an RTX 5070 Ti](../docs/images/desktop.png)

One window with the pipeline in-process:
- the captions: what was said, the emotion read from the voice (label and V·A·D), the Hindi as
  it is drafted and then finalized, and the ECS of the speech that played, with the emotion each
  clause sounded like;
- the latency budget, live: p50 and p95 per stage against the blueprint's budget, the
  translation queued to play, and dropouts;
- the microphone and speaker, and Listen / Play file / Stop.

Built with `-DEE_WITH_MINIAUDIO=ON -DEE_WITH_DESKTOP=ON`. It needs the X11 development headers
(apt: `libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`). Wayland desktops,
WSLg included, run it through XWayland.

```bash
emotionedge-desktop --config config/pipeline.engines.yaml        # then Listen, or Play file
emotionedge-desktop --input speech.wav --mute --screenshot shot.png  # render one frame and exit
```

Hindi needs text shaping: vowel signs reorder and consonants join into conjuncts. Dear ImGui
does not do that. So HarfBuzz 14.5.1 (MIT) shapes each caption, and the glyphs are rasterized
by index into a texture ImGui draws from (`apps/desktop/shaped_text.cpp`). Dear ImGui v1.92.9b
(MIT), GLFW 3.5.1 (zlib) and HarfBuzz are fetched at pinned, hash-verified versions. So is Noto
Sans Devanagari (SIL Open Font License 1.1), the font for both scripts; the build puts its
license next to it.

`--screenshot` renders into a hidden window and saves a frame, by default one second after the
session ends. That is how the UI is checked without a person at the screen. CI runs it under
Xvfb on the stand-in engines and keeps the image. The microphone path is the same as
`emotionedge live`; it was not exercised for this change.

## Planned (roadmap phase 5 "Ship")

- **Android and Jetson** builds.
