"""Speaker conditioning for the single-speaker AhsanTalal Matcha checkpoint WITHOUT reshaping any layer.

Matcha's own n_spks>1 path concatenates a speaker vector into the encoder and decoder inputs, which changes the
shape of ~6.2 M pretrained parameters. Instead, two zero-initialised embeddings are ADDED:
  emb_t  (n_spk, 1024)  added to the decoder's time embedding after time_mlp -> reaches every U-Net block
  emb_mu (n_spk, 80)    added to the encoder prior mu (in normalised mel space) -> speaker spectral envelope
The text encoder and the duration predictor get no speaker input: pronunciation and rhythm are shared.
At step 0 the output is exactly the base checkpoint's.

attach(model, n_spk)  monkey-patches forward/synthesise to take `spks` (speaker ids) through this path.
load_for_inference(ckpt, device, spk)  loads a trained checkpoint into matcha_torch's cache so
tts.synth renders speaker `spk`.

Speakers in the shipped model: 0 = the target voice (the kiosk voice), 1 = the Rasa Urdu studio voice
(support data only). The kiosk always speaks as speaker 0.
"""
import torch
from torch import nn


class SpkCond(nn.Module):
    """n_ch > 1 adds a recording-condition ("channel") switch, factorised from the speaker: a row's id is
    speaker + n_spk * channel, and the two embeddings are summed. Round 5: channel 0 = clean, 1 = noisy; the
    target voice is only ever heard noisy, Rasa both ways, so the channel vectors can only learn the noise and
    "target, clean" (id 0) is reachable at speaking time. (The shipped recipe uses n_ch = 1.)"""

    def __init__(self, n_spk, t_dim=1024, mel_dim=80, n_ch=1):
        super().__init__()
        self.n_spk, self.n_ch = n_spk, n_ch
        self.emb_t = nn.Embedding(n_spk, t_dim)
        self.emb_mu = nn.Embedding(n_spk, mel_dim)
        nn.init.zeros_(self.emb_t.weight)
        nn.init.zeros_(self.emb_mu.weight)
        if n_ch > 1:
            self.ch_t = nn.Embedding(n_ch, t_dim)
            self.ch_mu = nn.Embedding(n_ch, mel_dim)
            nn.init.zeros_(self.ch_t.weight)
            nn.init.zeros_(self.ch_mu.weight)

    def t(self, ids):
        out = self.emb_t(ids % self.n_spk)
        return out + self.ch_t(ids // self.n_spk) if self.n_ch > 1 else out

    def mu(self, ids):
        out = self.emb_mu(ids % self.n_spk)
        return out + self.ch_mu(ids // self.n_spk) if self.n_ch > 1 else out


def attach(model, n_spk, default_spk=None, n_ch=1):
    model.spk_cond = SpkCond(n_spk, n_ch=n_ch)
    model._cur_spk = None
    model._default_spk = default_spk

    enc_forward = model.encoder.forward

    def encoder_forward(x, x_lengths, spks=None):
        mu, logw, mask = enc_forward(x, x_lengths, None)
        if model._cur_spk is not None:
            mu = mu + model.spk_cond.mu(model._cur_spk).to(mu.dtype)[:, :, None] * mask
        return mu, logw, mask

    model.encoder.forward = encoder_forward

    tm = model.decoder.estimator.time_mlp
    tm_forward = tm.forward

    def time_mlp_forward(sample, condition=None):
        out = tm_forward(sample)
        if model._cur_spk is not None:
            out = out + model.spk_cond.t(model._cur_spk).to(out.dtype)
        return out

    tm.forward = time_mlp_forward

    fwd = model.forward

    def forward(x, x_lengths, y, y_lengths, spks=None, **kw):
        model._cur_spk = spks.long() if spks is not None else None
        return fwd(x, x_lengths, y, y_lengths, spks=None, **kw)

    model.forward = forward

    syn = model.synthesise

    def synthesise(x, x_lengths, n_timesteps, temperature=1.0, spks=None, length_scale=1.0):
        if spks is None and model._default_spk is not None:
            spks = torch.full((x.shape[0],), model._default_spk, dtype=torch.long, device=x.device)
        model._cur_spk = spks.long() if spks is not None else None
        return syn(x, x_lengths, n_timesteps, temperature, None, length_scale)

    model.synthesise = synthesise
    return model


def has_spk_cond(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    return any(k.startswith("spk_cond.") for k in sd)


def load_for_inference(ckpt_path, device, spk=0):
    """Populate matcha_torch's cache for ckpt_path with a speaker-conditioned model."""
    import matcha_torch
    c = matcha_torch._load(ckpt_path, device)
    model = c["model"]
    if getattr(model, "spk_cond", None) is None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        n = sd["spk_cond.emb_t.weight"].shape[0]
        n_ch = sd["spk_cond.ch_t.weight"].shape[0] if "spk_cond.ch_t.weight" in sd else 1
        attach(model, n, default_spk=spk, n_ch=n_ch)
        model.spk_cond.load_state_dict({k[len("spk_cond."):]: v for k, v in sd.items() if k.startswith("spk_cond.")})
        model.spk_cond.to(device)
    model._default_spk = spk
    return c
