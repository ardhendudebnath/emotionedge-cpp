"""Automatic judges of the emotion in speech, for evaluating emotion out (ml/README.md).

Three models, built by different labs on different data:
- **emotion2vec+** (emotion2vec/emotion2vec_plus_base): the runtime's consistency judge. The style
  offsets and prosody plans were trained against it, so it is the training judge.
- **audeering** (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim): V·A·D regression on
  MSP-Podcast. It took no part in training.
- **IEMOCAP** (speechbrain/emotion-recognition-wav2vec2-IEMOCAP, Apache-2.0): wav2vec2-base
  fine-tuned on IEMOCAP's acted English, four classes (neutral, anger, joy, sadness). The third
  judge, for where the other two disagree. It also took no part in training.

Each takes 16 kHz mono float audio. The models download into the Hugging Face cache on first use.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

from . import emotion_space as es

EMOTION2VEC = "emotion2vec/emotion2vec_plus_base"
AUDEERING = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
IEMOCAP = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
IEMOCAP_BASE = "facebook/wav2vec2-base"  # its architecture; only the config is fetched


class IemocapJudge:
    """SpeechBrain's IEMOCAP emotion classifier, run with transformers alone.

    The same computation as the model's own interface (custom_interface.py and SpeechBrain's
    wav2vec2 wrapper): layer norm over the waveform, wav2vec2-base, layer norm over the whole
    (time × channel) output, mean over time, a linear layer without bias, softmax. SpeechBrain
    adds ~1e-5 of random noise to the pooled mean; this leaves it out, so verdicts repeat.
    Built from wav2vec2-base's config and the model's own checkpoint, so wav2vec2-base's weights,
    which the checkpoint replaces anyway, are never downloaded.
    """

    def __init__(self, device: str = "cpu") -> None:
        import torch  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore
        from transformers import Wav2Vec2Config, Wav2Vec2Model  # type: ignore

        self.torch = torch
        self.device = device
        self.encoder = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(IEMOCAP_BASE))
        state = torch.load(hf_hub_download(IEMOCAP, "wav2vec2.ckpt"), map_location="cpu", weights_only=True)
        self.encoder.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
        self.encoder = self.encoder.to(device).eval()
        head = torch.load(hf_hub_download(IEMOCAP, "model.ckpt"), map_location="cpu", weights_only=True)
        self.head = head["0.w.weight"].to(device)  # (classes, 768)
        text = open(hf_hub_download(IEMOCAP, "label_encoder.txt"), encoding="utf-8").read()
        index = {int(i): lab for lab, i in re.findall(r"'(\w+)' => (\d+)", text) if lab != "starting_index"}
        self.labels = [es.DATASET_LABELS[index[i]] for i in range(len(index))]

    def probs(self, audio16) -> Dict[str, float]:
        """P(label) over neutral, anger, joy, sadness."""
        torch = self.torch
        import torch.nn.functional as F  # type: ignore

        x = torch.as_tensor(audio16, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            x = F.layer_norm(x, x.shape[1:])
            h = self.encoder(x).last_hidden_state
            h = F.layer_norm(h, h.shape[1:])
            p = torch.softmax(h.mean(dim=1) @ self.head.T, dim=-1)[0]
        return dict(zip(self.labels, p.tolist()))


class Emotion2vecJudge:
    """emotion2vec+ (the runtime's consistency judge), through FunASR."""

    def __init__(self, device: str = "cpu") -> None:
        from funasr import AutoModel  # type: ignore

        self.auto = AutoModel(model=EMOTION2VEC, hub="hf", disable_update=True, device=device)

    def probs(self, audio16) -> Dict[str, float]:
        """P(label) over its nine classes, named as the blueprint names them (angry -> anger, ...)."""
        res = self.auto.generate(audio16, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        return {es.CLASS_ALIASES.get(label.split("/")[-1], label.split("/")[-1]): float(p)
                for label, p in zip(res["labels"], res["scores"])}


class AudeeringJudge:
    """audeering's dimensional model: V·A·D, rescaled from [0, 1] to [-1, 1]."""

    def __init__(self, device: str = "cpu") -> None:
        import torch  # type: ignore
        from torch import nn  # type: ignore
        from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor  # type: ignore

        class RegressionHead(nn.Module):  # audeering's model card
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, x):
                return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                return self.classifier(self.wav2vec2(input_values)[0].mean(dim=1))

        self.torch = torch
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(AUDEERING)
        self.model = EmotionModel.from_pretrained(AUDEERING).to(device).eval()

    def vad(self, audio16) -> Tuple[float, float, float]:
        x = self.processor(audio16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with self.torch.no_grad():
            a, d, v = self.model(x)[0].tolist()  # the model's order: arousal, dominance, valence
        return (2 * v - 1, 2 * a - 1, 2 * d - 1)
