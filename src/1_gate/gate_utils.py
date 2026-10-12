"""Helpers for gate.py, copied unchanged from the files named below so their whole projects need not come along.

  Omni, load16, _semiglobal, urdu_cers   kaani/pk-r09-meter-reward-dpo/src/meter.py
      An independent Urdu check: Meta's Omnilingual ASR (EmreAkgul/omniASR-CTC-300M-ONNX, from the Hugging Face cache,
      CPU) transcribes a clip, and urdu_cers scores only the Urdu stretches of the transcript against it.
  Embedder                               kaani/pk-bilingual/src/dataset/spk_embed.py
      TitaNet-large speaker embeddings (nvidia/speakerverification_en_titanet_large, via NeMo). Unlike WavLM it can tell
      Pakistani women apart (equal-error 0% vs 28.5% on Anam vs others).
"""
import os
import re

import numpy as np
import soundfile as sf
import torch

LATIN = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
URDU_WORD = re.compile(r"[؀-ۿ]+")
OMNI = "EmreAkgul/omniASR-CTC-300M-ONNX"


class Omni:
    def __init__(self, threads=6):
        import onnxruntime as ort
        import sentencepiece as spm
        from huggingface_hub import snapshot_download
        p = snapshot_download(OMNI)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(os.path.join(p, "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.sp = spm.SentencePieceProcessor(model_file=os.path.join(p, "tokenizer.model"))

    def __call__(self, w16):
        w = ((w16 - w16.mean()) / (w16.std() + 1e-7)).astype("float32")
        ids = self.s.run(None, {"audio": w[None]})[0][0].argmax(-1)
        out, prev = [], -1
        for i in ids:
            if i != prev and i != 0:
                out.append(int(i))
            prev = i
        return self.sp.decode(out)


def load16(path):
    import librosa
    w, sr = sf.read(path, dtype="float32")
    return librosa.resample(w if w.ndim == 1 else w.mean(1), orig_sr=sr, target_sr=16000)


def _semiglobal(seg, hyp):
    """Fewest edits turning `seg` into some stretch of `hyp` (free start and end in hyp)."""
    prev = [0] * (len(hyp) + 1)
    for i, a in enumerate(seg, 1):
        cur = [i] + [0] * len(hyp)
        for j, b in enumerate(hyp, 1):
            cur[j] = min(prev[j - 1] + (a != b), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return min(prev)


def urdu_cers(rows, judged):
    """Per clip: edits / characters over the Urdu stretches of the text only. Each run of Urdu words between English
    words is matched to its own place in the transcript, so English words that Omnilingual writes in Urdu script
    cost nothing."""
    out = {}
    for r, j in zip(rows, judged):
        hyp = re.sub(r"\s+", " ", " ".join(URDU_WORD.findall(j["omni"])))
        edits = chars = 0
        for seg in LATIN.split(r["text"]):
            seg = " ".join(URDU_WORD.findall(seg))
            if len(seg) >= 4:
                edits += _semiglobal(seg, hyp)
                chars += len(seg)
        if chars >= 10:
            out[r["clip"]] = edits / chars
    return out


class Embedder:
    def __init__(self, dev=None):
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
        self.m = self.m.to(self.dev).eval()

    def embed(self, wavs, bs=32):
        """16 kHz float32 arrays -> L2-normalised embeddings (n, 192). Equal lengths batch; others go one by one."""
        out = []
        for i in range(0, len(wavs), bs):
            chunk = wavs[i:i + bs]
            lens = torch.tensor([len(w) for w in chunk], device=self.dev)
            x = torch.zeros(len(chunk), int(lens.max()), device=self.dev)
            for j, w in enumerate(chunk):
                x[j, :len(w)] = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
            with torch.inference_mode():
                _, e = self.m.forward(input_signal=x, input_signal_length=lens)
            out.append(e.float().cpu().numpy())
        e = np.concatenate(out) if out else np.zeros((0, 192), np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
