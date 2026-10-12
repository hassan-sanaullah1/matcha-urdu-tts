# The shipped voice's dataset: background removed, gate dropouts repaired

**Train on this folder.** It is a single-speaker Urdu/English speech dataset for fine-tuning
[`AhsanTalal/urdu-matcha-tts`](https://huggingface.co/AhsanTalal/urdu-matcha-tts)
(Matcha-TTS) toward the voice of the presenter of a YouTube channel.

It is identical to an earlier build of the same dataset (same clips, same text, same filelists) except for
one thing: the split-second dropouts in her voice have been repaired. See "What was fixed" below.

## At a glance

| | |
|---|---|
| **Usable audio** | **1.74 h** (864 clips) |
| Train / val | 838 clips (101.2 min) / 26 clips (3.1 min) |
| Split | Source-level: validation is one whole video (`-gI_ALIlkR8`), so no recording is in both splits |
| Clip length | 2.1–15.3 s, median 6.8 s |
| Audio format | WAV, 22050 Hz, mono, 16-bit PCM (matches the checkpoint) |
| Speaker | 1, verified per clip against an enrollment of her voice (speaker embedding ≥ 0.50) |
| Language mix | Urdu with English words: 84% of clips contain English, 0.2% are mostly English, 4% are pure Urdu |
| Background | Music and cartoon sound removed (MossFormer2_SE_48K); every clip DNSMOS background score ≥ 3.4 (a clean reference channel averages 3.88) |
| Edges | Every clip starts and ends in measured silence (median −71 dBFS start, −66 dBFS end) |
| Level | RMS −10.3 dBFS median, 2.1 dB spread between clips (p5–p95) |
| Validation | Passes `validate_dataset.py`: filelists parse, all wavs correct, all characters in the model's vocabulary, no split leakage |

## How it was built, and why

The channel's videos have a music bed plus sound from the AI cartoons she demonstrates, and
parts of the channel are narrated by a different voice. Each stage below fixes one measured problem.

1. **Only her voice.** 21 of the channel's uploads (3.18 h) are narrated by her. The other
   videos score 0.27–0.45 speaker similarity to her voice against her own 0.73–0.81, so they
   were excluded: the 15 parenting videos, the story shorts, the naats and the rhymes. An
   earlier, unfiltered build mixed both voices. Each clip is also checked (`not_target_speaker`).
2. **Background removed.** The earlier attempt used music separators (Demucs), which keep every
   voice, including the cartoon voices. They managed +2.9 dB. A speech enhancer trained with
   music as noise (ClearerVoice `MossFormer2_SE_48K`) gained +5 to +10 dB on this channel's real
   background, measured against ground truth. It was run on the original 48 kHz downloads and
   was approved by ear before use.
3. **Cuts only in real pauses.** Whisper's word timestamps are often 50–200 ms off, and the old
   pipeline cut clips inside words. Here a clip may only be cut inside a silence of at least
   100 ms measured in the audio. The silence must be 30 dB below loud speech overall and 40 dB
   below in the voice band (80–400 Hz), so breaths count as pauses and fading vowels never do.
   A clip whose first or last 30 ms is not silent is rejected.
4. **Other voices, leftover background.** Diarization (from the original audio) rejects clips
   that are less than 90% her. A DNSMOS background score below 3.4 rejects residual noise.
5. **Consistent spelling.** 36 words the pipeline could not settle were decided by hand. 27
   English loanwords are always written in English (`گوگل` → `google`). 9 real Urdu words that the
   English pass had misheard stay in Urdu. This fixed transcripts such as
   `comment میں pooch سکتے ہیں`, where she said `پوچھ`. See `config/force_english.txt` and
   `config/keep_urdu.txt`.
6. **Automatic checks as before:** Whisper confidence, independent re-transcription, a CTC
   acoustic model, clipping, speaking rate. Uncertain clips are held back, not guessed at.

## What was fixed in this copy: noise-gate dropouts

Her narration was recorded through a noise gate. Whenever her voice dips between syllables, the
gate mutes the track for about 25 ms, heard as the voice "stopping for a split second" and
resuming. This is in her original recording: her music-free original clips have 12.8 such gaps
per minute, while a reference channel's recordings have 0.1. The background music used to hide it.

A repair script fills each gap with a quiet continuation of the sound around it:
her own pitch periods where both sides are voiced, otherwise noise with the neighbouring
spectrum. The fill is set 19 dB below the quieter side, the level measured inside gaps in
ungated speech. No sample outside a gap is changed, and clip edges are untouched.

| Check | Result |
|---|---|
| Gaps repaired | 1,636 in 632 clips; 1 left |
| Ground truth (clean speech gated like hers, then repaired): level error inside gaps | 10.2 → **2.2 dB** |
| Ground truth: spectral error inside gaps | 55.2 → **12.1 dB** |
| Ground truth: real pauses wrongly filled | **0** of 94 |
| DNSMOS speech quality / UTMOS naturalness (139 of her clips) | 3.294 → 3.301 / 3.574 → 3.563 (no measurable change) |
| Whisper CER on the 50 most-repaired clips | 0.249 → 0.256 (within noise) |

`degate_report.csv` lists every clip's repairs. To undo any clip, copy it from the earlier build's `wav/` folder.

## Quick start (Matcha-TTS fine-tuning)

```bash
cd <this dataset's folder>
DS="$(pwd)"
sed "s#^wav/#$DS/wav/#" train.txt > train_abs.txt
sed "s#^wav/#$DS/wav/#" val.txt   > val_abs.txt
```

`configs/data/voice.yaml` in your Matcha-TTS checkout:

```yaml
_target_: matcha.data.text_mel_datamodule.TextMelDataModule
name: voice
train_filelist_path: /ABSOLUTE/PATH/<this dataset's folder>/train_abs.txt
valid_filelist_path: /ABSOLUTE/PATH/<this dataset's folder>/val_abs.txt
batch_size: 8            # 6 GB GPU; clips run to 15.3 s, lower this if you hit OOM
num_workers: 4
pin_memory: true
cleaners: [urdu_cleaners_graphemes]
add_blank: true
n_spks: 1
n_fft: 1024
n_feats: 80
sample_rate: 22050
hop_length: 256
win_length: 1024
f_min: 0
f_max: 8000
data_statistics:          # keep the CHECKPOINT's values when fine-tuning
  mel_mean: -1.1572901977914265
  mel_std: 2.2749757426509674
seed: 1234
load_durations: false
```

`urdu_cleaners_graphemes` must be registered in `matcha/text/cleaners.py`, as a pass-through of
graphemes that only collapses whitespace. Keep the checkpoint's `data_statistics`: the pretrained
decoder expects mels normalised with exactly these values.

### Fine-tuning notes for 1.74 h of one voice

- **Start from the checkpoint, never from scratch.** 1.74 h is plenty to move a pretrained
  model to a new voice, and far too little to train one.
- **Use a low learning rate** (about 1/5–1/10 of the from-scratch value) and watch validation
  loss. With 26 validation clips it is noisy, so judge by listening to fixed test sentences
  every few thousand steps.
- **Level:** her clips average −10 dBFS RMS, against −20.5 for a reference clean-studio voice,
  because her voice is heavily compressed at the source. That is fine on its own. If you mix this
  with another dataset, apply one uniform gain to one of them so their levels match.
- **Content is mostly Urdu with English terms** (YouTube, AI tools). That suits an Urdu model well.
- Clips auto-released from review are marked in `metadata.csv` (`promoted`, 36 clips).
  Filter them out if you want only clips that passed every check on their own.

## Files

| File | What it is | Train on it? |
|---|---|---|
| `train.txt`, `val.txt` | Filelists, `wav/clip.wav\|transcript` | **Yes** |
| `wav/` | The 864 clips | **Yes** |
| `degate_report.csv` | Gate repairs per clip | Reference |
| `metadata.csv` | Per clip: source video, timestamps, all check scores, split, `promoted` | Reference |
| `report.txt` / `report.json` | Build statistics | Reference |
| `codeswitch_ledger.json`, `numbers_ledger.json` | Spelling decisions | Reference |
| `needs_review.csv`, `wav_review/`, `review.txt` | 373 clips (47.0 min) held back as uncertain, mostly one unsure word | **No** |
| `rejected.csv` | 285 clips (35.9 min) that failed a hard check; no audio exported | **No** |

## Known limits

- **Transcripts were not checked by a human.** Automatic checks agree strongly, but no word
  error rate has been measured against a human listener.
- **The review pool (47 min) is recoverable by listening.** Most clips hold one uncertain word
  or a code-switch spelling question, not an audio problem.
- **Her greeting appears 4 times**, recorded in different videos. These are genuine separate
  recordings and harmless.

## Rebuilding

This describes rebuilding in the external dataset-building project (`~/Desktop/urdu-dataset-builder`), not in
kiosk-tts. From that project's root. Everything below is cached; nothing is re-transcribed unless forced.

```bash
# 1. clean the sources (separate venv, GPU, ~20 min)
PYTHONPATH=. venv_clean/bin/python music_noise_removal/second_attempt/clean_sources.py urls_voice_hervoice.txt
# 2. build
PYTHONPATH=. venv/bin/python run_pipeline.py --urls urls_voice_hervoice.txt --output dataset_voice_v2 \
  --no-local-audio --clean-audio-dir work/audio_22k_se --snap-edges \
  --target-speaker models/target_speakers/voice_her_wespeaker.npy --min-speaker-sim 0.5 \
  --min-dnsmos-bak 3.4 --cache-suffix se --diarization-from segments
# 3. repair gate dropouts -> dataset_voice_v2_fixed
PYTHONPATH=. venv/bin/python music_noise_removal/second_attempt/make_fixed_dataset.py dataset_voice_v2
# 4. check
PYTHONPATH=. venv/bin/python music_noise_removal/second_attempt/qa_dataset.py dataset_voice_v2_fixed
PYTHONPATH=. venv/bin/python validate_dataset.py --dataset dataset_voice_v2_fixed
```
