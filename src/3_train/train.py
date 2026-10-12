"""STEP 3 -- train: fine-tune the BASE model (models/base/matcha_urdu.ckpt, AhsanTalal/urdu-matcha-tts) on a voice.

Voice-agnostic: run it on the shipped voice's lists, or on any voice you built with step 1 + step 2.

What it does:
  - loads the base checkpoint (never an earlier fine-tune, so no earlier mistakes carry over);
  - starts each English phone symbol's untrained embedding row from the Urdu letter with the same sound
    (phone_init.INIT_FROM, with SEED_V2 below for the vowels when --seed-table v2);
  - with --support, adds two small speaker vectors (common/model/spk_cond.py): speaker 0 = the target voice
    (--data), speaker 1 = a support voice such as Rasa Urdu studio speech (--support), which keeps rare Urdu
    sounds from being forgotten on a small target dataset. --support is optional: leave it out to train on the
    target voice alone;
  - trains everything (encoder, duration predictor, decoder) and saves checkpoints/<tag>/step=*.ckpt.

Inputs:  data/3_training_lists/voice/{train,val}.txt, data/3_training_lists/rasa_female/train.txt
Outputs: checkpoints/<tag>/ (step=*.ckpt, args.json, train_ms.txt, val_ms.txt), logs/<tag>/ (loss curves)

The exact recipe (all flags explained) is in src/3_train/train.sh. Run from the kiosk-tts folder (the training
lists hold paths relative to it). Needs a CUDA GPU; 8,000 steps took about 41 minutes on a 6 GB card.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import argparse
import functools
import json
import time

import torch

import phones
from phone_init import INIT_FROM

# Round 5: INIT_FROM starts u, o and ɔ all from و, e and ɛ both from ے, and the length mark from ا, so "cooled"
# and "cold" (kuːlɖ / koːlɖ) begin as the same input and the model hears them merged. v2 seeds each vowel from how
# Urdu writes that sound (pesh+vao = uː, zabar+vao = ɔ, zabar+ye = ɛ, zer+ye = iː) and starts the length mark
# neutral (the blank, like the stress mark) instead of carrying an "aa". Caveat: the diacritics are nearly absent
# from the base's training text (13-33 per 1.5 h of Rasa transcripts), so their rows are close to untrained; a
# mean with a trained letter is still distinct from the letter alone, which is the point.
SEED_V2 = {
    "u": ("ُ", "و"), "o": "و", "ɔ": ("َ", "و"),
    "e": "ے", "ɛ": ("َ", "ے"),
    "i": ("ِ", "ی"), "a": "ا",
    "ː": "_",
}

ROOT = paths.ROOT
BASE_REPO = "AhsanTalal/urdu-matcha-tts"       # where models/base/ came from (recorded in args.json)
BASE_FILE = "matcha_urdu.ckpt"


PAUSE_DB = 32.0          # her noise floor sits ~35 dB below speech (24 Sep); pauses are within ~3 dB of it
PAUSE_FRAMES = 17        # 200 ms: longer than any stop closure or fricative, so no speech sound is ever masked
DB_PER_NAT = 8.686       # Matcha mels are natural-log magnitudes


def pause_mask(x1, mel_mean, mel_std, mask):
    """1 for frames to keep, 0 for long pauses. Judged against the window's own loud frames (95th percentile)."""
    e = (x1 * mel_std + mel_mean).mean(1) * DB_PER_NAT             # (b, t) mean log-mel level in dB
    keep = mask[:, 0].clone()
    for i in range(e.shape[0]):
        valid = mask[i, 0] > 0
        if valid.sum() < PAUSE_FRAMES:
            continue
        quiet = (e[i] < torch.quantile(e[i][valid], 0.95) - PAUSE_DB) & valid
        run = 0
        for j in range(e.shape[1] + 1):
            if j < e.shape[1] and quiet[j]:
                run += 1
                continue
            if run >= PAUSE_FRAMES:
                keep[i, j - run:j] = 0
            run = 0
    return keep[:, None]


def mask_pauses(model):
    """Unused experiment (Round 5 option A): the decoder is not trained to reproduce the room noise in the target
    voice's pauses (speaker 0 rows only). YouTube clips carry a noise floor well above a studio voice; speech frames,
    including the quietest consonants, stay fully trained, and alignment / duration / prior losses are untouched.
    The shipped recipe does not use this (no --mask-pauses)."""
    import torch.nn.functional as F
    dec = model.decoder
    stats = dict(kept=0.0, total=0.0)

    def compute_loss(x1, mask, mu, spks=None, cond=None):
        b = mu.shape[0]
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - dec.sigma_min) * t) * z + t * x1
        u = x1 - (1 - dec.sigma_min) * z
        pred = dec.estimator(y, mask, mu, t.squeeze(), spks)
        m = mask
        spk = getattr(model, "_cur_spk", None)
        if spk is not None:
            keep = pause_mask(x1.detach(), model.mel_mean, model.mel_std, mask)
            tgt = (spk == 0).to(mask.dtype)[:, None, None]      # speaker 0 = the target voice
            m = mask * (tgt * keep + (1 - tgt))
            stats["kept"] += float((m * tgt).sum())
            stats["total"] += float((mask * tgt).sum())
        loss = F.mse_loss(pred * m, u * m, reduction="sum") / (torch.sum(m) * u.shape[1])
        return loss, y

    dec.compute_loss = compute_loss
    model._pause_stats = stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=paths.VOICE_LISTS, help="folder with the target speaker's train.txt and val.txt")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    # 344 frames = 4.0 s of decoder context per clip (172 = 2.0 s left long sentences unfamiliar)
    ap.add_argument("--out-size", type=int, default=344)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--support", default="", help="support filelist (speaker 1), e.g. data/3_training_lists/rasa_female/train.txt")
    # --- the options from here to --mask-pauses were experiments; the shipped recipe uses only --no-row-b and --seed-table v2
    ap.add_argument("--support2", default="", help="second support voice (speaker 2), e.g. data/rasa_new/train.txt")
    ap.add_argument("--support2-share", type=float, default=0.5, help="share of the support rows given to speaker 2")
    ap.add_argument("--target-weight", type=float, default=0.0,
                    help="share of training rows that must come from the target speaker (0 = use both as they are)")
    ap.add_argument("--no-row-b", action="store_true", help="keep only row A (Urdu script + English phones)")
    ap.add_argument("--seed-table", choices=("v1", "v2"), default="v1",
                    help="v1: urdu-phones INIT_FROM; v2: contrasting vowels start apart (SEED_V2)")
    ap.add_argument("--noise-channel", action="store_true",
                    help="unused experiment: clean/noisy switch (target noisy, Rasa clean, Rasa+room-noise). Speak clean. "
                         "The shipped recipe does not use this")
    ap.add_argument("--mask-pauses", action="store_true",
                    help="unused experiment: decoder loss ignores the target's long pauses (room noise only); speech "
                         "frames always count")
    ap.add_argument("--spk-lr-mult", type=float, default=10.0,
                    help="learning-rate multiplier for the speaker tags only (at 1x the pilot's tags barely "
                         "separated: speaker 0 and 1 came out ~7 Hz apart)")
    a = ap.parse_args()

    import matcha_torch
    matcha_torch._prepare_imports()
    from lightning import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from matcha.data.text_mel_datamodule import TextMelDataModule
    from matcha.models.baselightningmodule import BaseLightningClass
    from matcha.models.matcha_tts import MatchaTTS

    BaseLightningClass.on_validation_end = lambda self: None

    base_path = paths.BASE_CKPT                    # always the untouched base model, never a fine-tuned one
    ck = torch.load(base_path, map_location="cpu", weights_only=False)
    hp = dict(ck["hyper_parameters"])
    stats = hp["data_statistics"]
    hp["n_vocab"] = ck["state_dict"]["encoder.emb.weight"].shape[0]
    hp["out_size"] = a.out_size
    hp["optimizer"] = functools.partial(torch.optim.Adam, lr=a.lr, weight_decay=0.0)
    hp["scheduler"] = None
    model = MatchaTTS(**hp)
    model.load_state_dict(ck["state_dict"], strict=True)
    print(f"loaded BASE {BASE_REPO}/{BASE_FILE} (strict) from {base_path}")

    vocab = json.load(open(paths.BASE_VOCAB, encoding="utf-8"))
    chars = sorted(set("".join(phones.PHONES + [phones.STRESS])))
    emb = model.encoder.emb.weight
    table = {**INIT_FROM, **SEED_V2} if a.seed_table == "v2" else INIT_FROM
    with torch.no_grad():
        src = {c: [emb[vocab[x]].clone() for x in (table[c] if isinstance(table[c], tuple) else (table[c],))]
               for c in chars}
        for c in chars:
            emb[vocab[c]] = torch.stack(src[c]).mean(0)
    print(f"initialised {len(chars)} phone rows from their Urdu letter rows (base weights), seed table {a.seed_table}")
    n_spks = 1
    train_list = os.path.join(a.data, "train.txt")
    val_list = os.path.join(a.data, "val.txt")
    if a.support:
        import spk_cond
        if a.noise_channel:
            spk_cond.attach(model, 2, n_ch=2)             # ids: speaker + 2 * channel (0 clean, 1 noisy)
            n_spks = 4
            ID_T, ID_S, ID_S2 = 2, 1, 3
        else:
            n_spks = 3 if a.support2 else 2
            spk_cond.attach(model, n_spks)
            ID_T, ID_S, ID_S2 = 0, 1, 2
        out_lists = os.path.join(ROOT, "checkpoints", a.tag)
        os.makedirs(out_lists, exist_ok=True)

        def tag_rows(path, spk):
            lines = [l for l in open(path, encoding="utf-8") if "|" in l]
            if a.no_row_b:
                # build_voice writes row A then row B per clip, so this keeps row A for the target voice. build_rasa
                # shuffles its lines, so for Rasa this keeps a random half of A and B rows (the recipe's default).
                lines = lines[::2]
            return [f"{l.split('|', 1)[0]}|{spk}|{l.rstrip().split('|', 1)[1]}\n" for l in lines]
        tgt, sup = tag_rows(train_list, ID_T), tag_rows(a.support, ID_S)
        import random as _r

        def fit(rows, want):
            return (rows * (want // len(rows) + 1))[:want] if want > len(rows) else _r.Random(1).sample(rows, want)
        if a.target_weight:
            # Repeat or trim the support speaker(s) so the target keeps the intended share of every epoch.
            want = int(len(tgt) * (1 - a.target_weight) / a.target_weight)
            if a.support2:
                w2 = int(want * a.support2_share)
                sup = fit(sup, want - w2) + fit(tag_rows(a.support2, ID_S2), w2)
            else:
                sup = fit(sup, want)
        elif a.support2:
            sup = sup + tag_rows(a.support2, ID_S2)
        rows = tgt + sup
        _r.Random(0).shuffle(rows)
        train_list = os.path.join(out_lists, "train_ms.txt")
        open(train_list, "w", encoding="utf-8").writelines(rows)
        val_list = os.path.join(out_lists, "val_ms.txt")
        open(val_list, "w", encoding="utf-8").writelines(tag_rows(os.path.join(a.data, "val.txt"), ID_T))
        print(f"multi-speaker: target rows {len(tgt)} (id {ID_T}), support rows {len(sup)} "
              f"(id {ID_S}: {sum(1 for l in sup if l.split('|')[1] == str(ID_S))}, id {ID_S2}: {sum(1 for l in sup if l.split('|')[1] == str(ID_S2))}); "
              "zero-init speaker vectors")
    for p in model.parameters():
        p.requires_grad = True
    print(f"training all {sum(p.numel() for p in model.parameters()):,} parameters")
    if n_spks > 1:
        spk_p = [p for n, p in model.named_parameters() if n.startswith("spk_cond.")]
        rest = [p for n, p in model.named_parameters() if not n.startswith("spk_cond.")]
        model.configure_optimizers = lambda: torch.optim.Adam(
            [{"params": rest, "lr": a.lr}, {"params": spk_p, "lr": a.lr * a.spk_lr_mult}], weight_decay=0.0)
        print(f"speaker tags at {a.spk_lr_mult:g}x learning rate")

    dm = TextMelDataModule(
        name=a.tag, train_filelist_path=train_list,
        valid_filelist_path=val_list,
        batch_size=a.bs, num_workers=a.workers, pin_memory=True,
        cleaners=["urdu_cleaners_graphemes"], add_blank=True, n_spks=n_spks,
        n_fft=1024, n_feats=80, sample_rate=22050, hop_length=256, win_length=1024,
        f_min=0, f_max=8000, data_statistics=stats, seed=1234, load_durations=False,
    )
    out = os.path.join(ROOT, "checkpoints", a.tag)
    os.makedirs(out, exist_ok=True)
    json.dump({**vars(a), "base": f"{BASE_REPO}/{BASE_FILE}"}, open(os.path.join(out, "args.json"), "w"), indent=1)
    ckpt_cb = ModelCheckpoint(dirpath=out, filename="{step:06d}", every_n_train_steps=a.ckpt_every,
                              save_top_k=-1, save_last=False, save_weights_only=True)
    if a.mask_pauses:
        mask_pauses(model)
        print(f"decoder loss: target pauses masked (>= {PAUSE_DB} dB below the window's loud frames for >= {PAUSE_FRAMES} frames)")
    trainer = Trainer(
        max_steps=a.steps, accelerator="gpu", devices=1, precision="16-mixed",
        accumulate_grad_batches=a.accum, gradient_clip_val=5.0, callbacks=[ckpt_cb],
        logger=CSVLogger(os.path.join(ROOT, "logs"), name=a.tag), check_val_every_n_epoch=None,
        val_check_interval=a.ckpt_every, limit_val_batches=8, log_every_n_steps=50,
        enable_progress_bar=False,
    )
    t0 = time.time()
    trainer.fit(model, dm)
    print(f"done -> {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
