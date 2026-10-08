#!/usr/bin/env python3
"""Reference client for `emotionedge serve`: streams a WAV file at speaking pace, prints the live
captions, saves the translated speech, and reports the latency a client sees.

    python apps/web/stream_wav.py speech.wav [--url ws://127.0.0.1:8080/] [--out translated.wav]
        [--token TOKEN] [--ca ca.pem]   # a server started with --token / with TLS (wss://)

Latency here is from the moment the client has sent the end of an utterance (its transcript's
`end` time, at the pace sent) to the first translated audio the client receives after that
utterance's translation. It includes the server's playback pacing and the network. It is only
measured when no earlier translation was still playing; otherwise that audio would be counted.
The server's own end-to-end figures (from the "done" event) are printed too. The input must be
16-bit PCM; requires `pip install websockets`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:8080/")
    parser.add_argument("--out", type=Path, default=Path("translated.wav"))
    parser.add_argument("--token", help="sent as 'Authorization: Bearer <token>'")
    parser.add_argument("--ca", type=Path, help="CA certificate to trust for wss:// (default: the system's)")
    args = parser.parse_args()
    import ssl

    import websockets  # type: ignore

    with wave.open(str(args.wav)) as w:
        if w.getsampwidth() != 2:
            raise SystemExit("the input must be 16-bit PCM")
        rate, channels = w.getframerate(), w.getnchannels()
        frames = w.readframes(w.getnframes())
    if channels > 1:  # keep the first channel
        width = 2 * channels
        frames = b"".join(frames[i:i + 2] for i in range(0, len(frames), width))

    url = args.url + ("&" if "?" in args.url else "?") + f"rate={rate}"
    events, audio, first_audio_after, queued = [], bytearray(), {}, []
    connecting = time.monotonic()
    options = {"max_size": None}
    if args.token:
        options["additional_headers"] = {"Authorization": f"Bearer {args.token}"}
    if url.startswith("wss://"):
        options["ssl"] = ssl.create_default_context(cafile=str(args.ca) if args.ca else None)
    async with websockets.connect(url, **options) as ws:
        ready = json.loads(await ws.recv())
        if ready["type"] != "ready":
            raise SystemExit(f"server: {ready}")
        output_rate = ready["output_rate"]
        print(f"ready {time.monotonic() - connecting:.2f} s after connecting: {ready['source_language']} -> "
              f"{ready['target_language']}, {rate} Hz in, {output_rate} Hz out")
        start = time.monotonic()

        async def send() -> None:
            block = rate // 50 * 2  # 20 ms of 16-bit samples
            for i, pos in enumerate(range(0, len(frames), block)):
                await asyncio.sleep(max(0.0, start + i * 0.02 - time.monotonic()))
                await ws.send(frames[pos:pos + block])
            await ws.send(json.dumps({"type": "end"}))

        async def receive() -> None:
            pending = []  # utterances translated while no audio was playing
            last_audio = -1.0
            async for message in ws:
                now = time.monotonic() - start
                if isinstance(message, bytes):
                    audio.extend(message)
                    last_audio = now
                    for utt in pending:
                        first_audio_after[utt] = now
                    pending.clear()
                    continue
                e = json.loads(message)
                e["_t"] = now
                events.append(e)
                if e["type"] == "transcript" and e["final"]:
                    print(f"{now:6.2f}s  heard       {e['text']}")
                elif e["type"] == "emotion":
                    print(f"{now:6.2f}s  felt        {e['label']} (V {e['valence']:+.2f} A {e['arousal']:+.2f} "
                          f"D {e['dominance']:+.2f}, confidence {e['confidence']:.2f})")
                elif e["type"] == "translation" and e["final"]:
                    print(f"{now:6.2f}s  translated  {e['text']}")
                    # Audio arrives every 10 ms while a translation plays.
                    (pending if now - last_audio > 0.15 else queued).append(e["utterance"])
                elif e["type"] == "error":
                    print(f"{now:6.2f}s  error       {e['message']}")
                elif e["type"] == "done":
                    return

        await asyncio.gather(send(), receive())

    with wave.open(str(args.out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(output_rate)
        w.writeframes(bytes(audio))
    print(f"wrote {args.out}: {len(audio) / 2 / output_rate:.1f} s of translated speech")

    latencies = []
    for e in events:
        if e["type"] == "transcript" and e["final"] and e["utterance"] in first_audio_after:
            latencies.append(1000.0 * (first_audio_after[e["utterance"]] - e["end"]))
    if latencies:
        latencies.sort()
        print(f"latency, end of speech -> first translated audio at the client: "
              f"{', '.join(f'{x:.0f}' for x in latencies)} ms"
              + (f" ({len(queued)} utterance(s) queued behind earlier speech, not measured)" if queued else ""))
    done = next((e for e in events if e["type"] == "done"), None)
    if done and done["session"].get("end_to_end", {}).get("count"):
        e2e = done["session"]["end_to_end"]
        print(f"server end to end: p50 {e2e['p50_ms']:.0f} ms, p95 {e2e['p95_ms']:.0f} ms over {e2e['count']} utterance(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
