#!/usr/bin/env python3
"""P4 "Export": Kokoro-82M (StyleTTS2 family, 24 kHz, blueprint 4.2) with explicit prosody
controls, as the voice directory core/tts/kokoro_engine.cpp loads:

    model.onnx    inputs  input_ids int64 [1, T]   phoneme ids, 0 at both ends
                          style     f32   [1, 256] voicepack row: 128 acoustic + 128 prosodic
                          speed     f32   [1]      global rate (>1 faster)
                          dur_scale f32   [T]      per-token duration multiplier (emphasis)
                          accent    f32   [T]      per-token pitch multiplier (pitch accents)
                          pitch_st  f32   [1]      pitch-mean shift, semitones
                          range     f32   [1]      pitch-range scale around the clause mean
                          fall      f32   [1]      final contour: +1 falls ~25 % over the last 30 %
                  outputs audio f32 [samples], durations int64 [T] (frames per token)
    config.json   Kokoro's config (phoneme vocabulary)
    voices/*.bin  voicepacks, float32 [510, 256]
    voices.json   per voice: gender and ECAPA-TDNN voice print (for matching the speaker)

The controls act between Kokoro's prosody predictor and its decoder: they scale the predicted
durations and edit the predicted F0 curve on voiced frames. They leave timbre (the acoustic
style) untouched. With neutral controls the output matches Kokoro's own forward pass, which
the script checks.

    python ml/export/export_kokoro_onnx.py --out ~/ee-models/tts/kokoro-82m-hi \\
        --espeak-lib ~/ee-deps/root/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1 \\
        --golden tests/golden/kokoro_g2p_hi.json

`--espeak-lib` points misaki's phonemizer at the espeak-ng the C++ runtime uses, so the golden
phonemes test the C++ port rather than a version difference between two espeak builds.

Requires: torch, kokoro, misaki, phonemizer-fork, speechbrain, onnx, onnxruntime.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

REPO = "hexgrad/Kokoro-82M"
VOICES = {"hf_alpha": "female", "hf_beta": "female", "hm_omega": "male", "hm_psi": "male"}
# Reference sentence for each voice's print, and the G2P golden set (punctuation, danda,
# numbers, code-mixed Latin words, quotes and parentheses).
REFERENCE = "नमस्ते, आज मौसम बहुत अच्छा है और हम सब मिलकर बाहर घूमने जा रहे हैं।"
G2P_SENTENCES = [
    REFERENCE,
    "मुझे यकीन नहीं हो रहा कि तुमने ऐसा किया!",
    "मुझे बहुत खेद है, मेरा इरादा तुम्हें दुख पहुँचाने का नहीं था।",
    "यह तो कमाल है, बहुत-बहुत धन्यवाद!",
    "और इसलिए मेरे अमेरिकी साथियों.",
    "मत पूछो!",
    "आपका देश क्या आपके लिए कर सकता है?",
    "पूछो कि तुम अपने देश के लिए क्या कर सकते हो।",
    "क्या तुम कल 5 बजे आ सकते हो?",
    "मैंने कहा, \"रुको\" (लेकिन उसने नहीं सुना)।",
    "मेरा फ़ोन नंबर 98765 है।",
    "यह email मुझे office से मिला।",
    "हाँ... शायद; पता नहीं: देखते हैं।",
    "ओह नहीं!",
    "ठीक है।",
]


def exact_stft_module(n_fft: int, hop: int):
    """An ONNX-exportable STFT/iSTFT (conv1d, no complex tensors) that reproduces torch.stft /
    torch.istft (center=True, periodic Hann). Kokoro's own CustomSTFT approximates both ends:
    - its inverse skips the doubling of bins 1..N/2-1 that a real inverse FFT needs;
    - its inverse skips the window-envelope normalization;
    - its forward pads by replicating instead of reflecting.
    With Kokoro's 20-point FFT those approximations tilt the output spectrum by about ±2 dB."""
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    class ExactStft(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_fft, self.hop = n_fft, hop
            bins = n_fft // 2 + 1
            window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64).numpy()
            n, k = np.arange(n_fft), np.arange(bins)
            angle = 2 * np.pi * np.outer(k, n) / n_fft  # [bins, n_fft]
            self.register_buffer("fwd_re", torch.tensor(np.cos(angle) * window, dtype=torch.float32).unsqueeze(1))
            self.register_buffer("fwd_im", torch.tensor(-np.sin(angle) * window, dtype=torch.float32).unsqueeze(1))
            # Real inverse DFT: x[n] = (1/N) [X0 + 2 sum_{0<k<N/2} Re(X_k e^{i..}) + X_{N/2} (-1)^n].
            scale = np.full(bins, 2.0 / n_fft)
            scale[0] = 1.0 / n_fft
            if n_fft % 2 == 0:
                scale[-1] = 1.0 / n_fft
            inv = scale[:, None] * window[None, :]
            self.register_buffer("inv_re", torch.tensor(np.cos(angle) * inv, dtype=torch.float32).unsqueeze(1))
            self.register_buffer("inv_im", torch.tensor(np.sin(angle) * inv, dtype=torch.float32).unsqueeze(1))
            self.register_buffer("win_sq", torch.tensor(window ** 2, dtype=torch.float32).view(1, 1, -1))

        def transform(self, x):  # x [B, T] -> magnitude, phase [B, bins, frames]
            p = self.n_fft // 2
            x = torch.cat([x[..., 1:p + 1].flip(-1), x, x[..., -p - 1:-1].flip(-1)], dim=-1)  # reflect
            x = x.unsqueeze(1)
            re = F.conv1d(x, self.fwd_re, stride=self.hop)
            im = F.conv1d(x, self.fwd_im, stride=self.hop)
            return torch.sqrt(re * re + im * im + 1e-14), torch.atan2(im, re)

        def inverse(self, magnitude, phase):  # -> [B, 1, T]
            y = (F.conv_transpose1d(magnitude * torch.cos(phase), self.inv_re, stride=self.hop)
                 - F.conv_transpose1d(magnitude * torch.sin(phase), self.inv_im, stride=self.hop))
            ones = torch.ones_like(magnitude[:, :1, :])
            envelope = F.conv_transpose1d(ones, self.win_sq, stride=self.hop)
            y = y / envelope.clamp(min=1e-11)
            p = self.n_fft // 2
            return y[..., p:-p]

    return ExactStft()


def check_exact_stft(module, n_fft: int, hop: int) -> None:
    import torch  # type: ignore

    x = torch.randn(1, 4000)
    window = torch.hann_window(n_fft, periodic=True)
    ref = torch.stft(x, n_fft, hop, n_fft, window=window, return_complex=True)
    mag, ph = module.transform(x)
    spec_err = float((torch.polar(mag, ph) - ref).abs().max())
    rec = module.inverse(mag, ph).squeeze(1)
    ref_rec = torch.istft(ref, n_fft, hop, n_fft, window=window)
    n = min(rec.shape[-1], ref_rec.shape[-1])
    inv_err = float((rec[..., :n] - ref_rec[..., :n]).abs().max())
    print(f"exact STFT vs torch.stft: max error {spec_err:.1e}; inverse vs torch.istft: {inv_err:.1e}")
    if spec_err > 1e-3 or inv_err > 1e-4:
        raise SystemExit("the exportable STFT does not match torch")


def main() -> int:
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore
    import torch  # type: ignore
    from huggingface_hub import hf_hub_download  # type: ignore

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--espeak-lib", type=Path, help="libespeak-ng the C++ runtime uses")
    parser.add_argument("--espeak-data", type=Path, help="its espeak-ng-data (default: next to the library)")
    parser.add_argument("--golden", type=Path, help="write G2P golden phonemes for the C++ test here")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    from misaki import espeak as misaki_espeak  # type: ignore  (sets espeakng_loader paths on import)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper  # type: ignore

    if args.espeak_lib:
        EspeakWrapper.set_library(str(args.espeak_lib))
        EspeakWrapper.set_data_path(str(args.espeak_data or args.espeak_lib.parent / "espeak-ng-data"))
    g2p = misaki_espeak.EspeakG2P(language="hi")

    from kokoro import KModel  # type: ignore

    # ONNX has no complex tensors. Export with an exact conv-based STFT in place of Kokoro's
    # approximate one, and check it against the standard (complex iSTFT) model below.
    kmodel = KModel(repo_id=REPO, disable_complex=True).eval()
    generator = kmodel.decoder.generator
    n_fft, hop = generator.stft.filter_length, generator.stft.hop_length
    generator.stft = exact_stft_module(n_fft, hop)
    check_exact_stft(generator.stft, n_fft, hop)
    reference_model = KModel(repo_id=REPO).eval()

    class KokoroControlled(torch.nn.Module):
        """Kokoro's forward_with_tokens with prosody controls between predictor and decoder."""

        def __init__(self, m) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids, style, speed, dur_scale, accent, pitch_st, range_scale, fall):
            m = self.m
            t = input_ids.shape[1]
            input_lengths = torch.full((1,), t, dtype=torch.long)
            text_mask = torch.zeros((1, t), dtype=torch.bool)  # batch of one: nothing padded
            bert_dur = m.bert(input_ids, attention_mask=(~text_mask).int())
            d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
            s = style[:, 128:]
            d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = m.predictor.lstm(d)
            duration = torch.sigmoid(m.predictor.duration_proj(x)).sum(dim=-1) / speed  # [1, T]
            duration = duration * dur_scale.unsqueeze(0)
            pred_dur = torch.round(duration).clamp(min=1).long().squeeze(0)  # [T]
            frame_token = torch.repeat_interleave(torch.arange(t), pred_dur)  # [F]
            aln = (torch.arange(t).unsqueeze(1) == frame_token.unsqueeze(0)).float().unsqueeze(0)  # [1, T, F]
            en = d.transpose(-1, -2) @ aln
            f0, n = m.predictor.F0Ntrain(en, s)  # [1, F0 frames]
            # Per-token accents spread over each token's frames, then onto the F0 rate.
            acc = (accent.view(1, 1, t) @ aln)  # [1, 1, F]
            acc = torch.nn.functional.interpolate(acc, size=f0.shape[-1], mode="nearest").squeeze(1)
            voiced = (f0 > 10.0).float()
            mean = (f0 * voiced).sum() / voiced.sum().clamp(min=1.0)
            pos = torch.arange(f0.shape[-1], dtype=torch.float32) / f0.shape[-1]
            ramp = ((pos - 0.7) / 0.3).clamp(0.0, 1.0)
            shaped = (mean * torch.pow(2.0, pitch_st / 12.0) + (f0 - mean) * range_scale) * acc
            shaped = shaped * (1.0 - 0.25 * fall * ramp)
            f0 = voiced * shaped.clamp(min=10.0) + (1.0 - voiced) * f0  # stays voiced
            t_en = m.text_encoder(input_ids, input_lengths, text_mask)
            asr = t_en @ aln
            audio = m.decoder(asr, f0, n, style[:, :128]).squeeze()
            return audio, pred_dur

    controlled = KokoroControlled(kmodel).eval()

    def phonemes_to_ids(ps: str):
        ids = [kmodel.vocab[c] for c in ps if c in kmodel.vocab]
        return torch.LongTensor([[0, *ids, 0]])

    def neutral(t):
        one = torch.ones(1)
        return (one, torch.ones(t), torch.ones(t), torch.zeros(1), one, torch.zeros(1))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "voices").mkdir(exist_ok=True)
    packs = {}
    for name in VOICES:
        pack = torch.load(hf_hub_download(REPO, f"voices/{name}.pt"), weights_only=True).squeeze(1).float()
        packs[name] = pack  # [510, 256]
        pack.numpy().astype("float32").tofile(args.out / "voices" / f"{name}.bin")

    ps, _ = g2p(REFERENCE)
    ids = phonemes_to_ids(ps)
    style = packs["hf_alpha"][len(ps) - 1].unsqueeze(0)
    with torch.no_grad():
        # The decoder's excitation adds random noise: seed the passes identically.
        speed, dur, acc, pst, rng, fall = neutral(ids.shape[1])
        torch.manual_seed(0)
        same_stft, _ = kmodel.forward_with_tokens(ids, style, 1.0)
        torch.manual_seed(0)
        ours, _ = controlled(ids, style, speed, dur, acc, pst, rng, fall)
        torch.manual_seed(0)
        complex_audio, _ = reference_model.forward_with_tokens(ids, style, 1.0)
    # A: the controls are transparent when neutral. Rounding in the re-derived F0 accumulates
    # through the sine generator's phase, so compare relative RMS error, not the largest sample.
    rel = float((same_stft - ours).pow(2).mean().sqrt() / same_stft.pow(2).mean().sqrt())
    print(f"neutral controls vs Kokoro forward: relative RMS error {rel:.2e} over {len(ours)} samples")
    if rel > 0.01:
        raise SystemExit("controlled forward pass disagrees with Kokoro")

    # B: the conv-based STFT sounds like the complex one. A sub-frame offset makes waveforms
    # incomparable, so compare log-mel spectra.
    import torchaudio  # type: ignore

    mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80)

    def log_mel(a):
        return 10 * torch.log10(mel(a) + 1e-6)

    n = min(len(ours), len(complex_audio))
    mel_db = float((log_mel(ours[:n]) - log_mel(complex_audio[:n])).abs().mean())
    print(f"exact conv STFT vs complex STFT: mean |log-mel difference| {mel_db:.2f} dB")
    if mel_db > 1.0:
        raise SystemExit("the ONNX-friendly STFT changes the sound")

    onnx_path = args.out / "model.onnx"
    names = ["input_ids", "style", "speed", "dur_scale", "accent", "pitch_st", "range", "fall"]
    with torch.no_grad():
        torch.onnx.export(controlled, (ids, style, speed, dur, acc, pst, rng, fall), str(onnx_path),
                          input_names=names, output_names=["audio", "durations"],
                          dynamic_axes={"input_ids": {1: "tokens"}, "dur_scale": {0: "tokens"},
                                        "accent": {0: "tokens"}, "audio": {0: "samples"},
                                        "durations": {0: "tokens"}},
                          opset_version=args.opset, dynamo=False)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def run_onnx(ids_, style_, speed_=1.0, dur_=None, acc_=None, pitch=0.0, range_=1.0, fall_=0.0):
        t = ids_.shape[1]
        feeds = {"input_ids": ids_.numpy(), "style": style_.numpy(), "speed": np.array([speed_], np.float32),
                 "dur_scale": np.ones(t, np.float32) if dur_ is None else dur_,
                 "accent": np.ones(t, np.float32) if acc_ is None else acc_,
                 "pitch_st": np.array([pitch], np.float32), "range": np.array([range_], np.float32),
                 "fall": np.array([fall_], np.float32)}
        return session.run(None, feeds)

    start = time.perf_counter()
    onnx_audio, _ = run_onnx(ids, style)
    ms = (time.perf_counter() - start) * 1000
    print(f"onnx vs torch: max |diff| {np.abs(onnx_audio - ours.numpy()).max():.2e}; "
          f"{ms:.0f} ms for {len(onnx_audio) / 24000:.2f} s of audio")

    # The controls must do what they say: pitch shift moves the median F0, speed changes length.
    import librosa  # type: ignore  (speechbrain / funasr pull it in)

    def median_f0(audio):
        f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=500, sr=24000)
        return float(np.nanmedian(f0[voiced])) if voiced.any() else float("nan")

    base_f0 = median_f0(onnx_audio)
    up, _ = run_onnx(ids, style, pitch=3.0)
    fast, _ = run_onnx(ids, style, speed_=1.25)
    print(f"control check: median F0 {base_f0:.0f} Hz -> {median_f0(up):.0f} Hz at +3 st "
          f"(expect x{2 ** (3 / 12):.3f}); length {len(onnx_audio)} -> {len(fast)} samples at speed 1.25")

    # Voice prints: ECAPA-TDNN (SpeechBrain) on each voice's reference sentence.
    from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=str(Path.home() / "ee-ml" / "sb-ecapa"),
                                           run_opts={"device": "cpu"})
    voices = {}
    for name, gender in VOICES.items():
        audio, _ = run_onnx(ids, packs[name][len(ps) - 1].unsqueeze(0))
        wav16 = torchaudio.functional.resample(torch.from_numpy(audio), 24000, 16000).unsqueeze(0)
        emb = ecapa.encode_batch(wav16).squeeze().numpy()
        voices[name] = {"gender": gender, "ecapa": [round(float(v), 6) for v in emb / np.linalg.norm(emb)],
                        "median_f0": round(median_f0(audio), 1)}
        print(f"  {name}: {gender}, median F0 {voices[name]['median_f0']} Hz")
    (args.out / "voices.json").write_text(json.dumps(voices) + "\n", encoding="utf-8")
    shutil.copy(hf_hub_download(REPO, "config.json"), args.out / "config.json")

    if args.golden:
        golden = [{"text": s, "phonemes": g2p(s)[0]} for s in G2P_SENTENCES]
        args.golden.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        for g in golden[:4]:
            print(f"  {g['text']}  ->  {g['phonemes']}")
        print(f"wrote {args.golden}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
