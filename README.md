# kiosk-tts — fine-tune Matcha-TTS into a Pakistani Urdu voice

A self-contained project that fine-tunes **Matcha-TTS** into a Pakistani voice that speaks Urdu, English and
code-switched text — originally built for the MARI Energies kiosk, but the pipeline itself is **generic**: it
holds everything needed to **use the shipped voice**, to **rebuild it from scratch**, and to **fine-tune a voice
of your own** on your own dataset.

- the code for every step, numbered in the order it runs (`src/1_gate` … `src/6_speak`), each one voice-agnostic;
- the base model to fine-tune from, and the vocoder;
- the shipped voice's dataset, its gated clips and the exact training lists;
- the **Rasa** support data (a studio Urdu voice trained alongside it — see
  ["Why Rasa is here"](#why-rasa-is-here) — optional for a voice of your own);
- the shipped, trained checkpoints, the evaluation tools and the frozen test sets.

The voice is **Matcha-TTS fine-tuned from the Urdu base model
[AhsanTalal/urdu-matcha-tts](https://huggingface.co/AhsanTalal/urdu-matcha-tts)**, with the Vocos 22 kHz vocoder.
The shipped checkpoint is `checkpoints/voice_demuffled/model.ckpt` (8,000 steps, then a de-muffle weight edit).
It was copied out of `~/Desktop/kaani` (26 rounds of experiments) and depends on none of it.

## Quick start — speak

```bash
cd ~/Desktop/kiosk-tts
PY=~/Desktop/kaani/envs/venv/bin/python          # or your own environment, see "Setup"

# speak one sentence (uses checkpoints/voice_demuffled)
$PY src/6_speak/speak.py "آج کی meeting میں ہم Mari Energies کے results discuss کریں گے۔" -o out.wav

# type-and-listen web page  ->  http://127.0.0.1:8024
$PY demo/server.py
```

`results/sample_voice.wav` is that first sentence already rendered. To hear the voice on many sentences, and
next to its speaker's real recordings, open **`samples/index.html`**.

Speaking needs no internet and nothing outside this folder, except the `espeak-ng` program (see Setup).

## The pipeline

```
 A speech dataset (built outside kiosk-tts; the shipped one lives in data/1_source/voice)
        │
 STEP 1  src/1_gate (or import_voice.py)   keep only clips that are clearly one speaker with clean Urdu
                                            ─► data/2_clean/<name>
 STEP 2  src/2_build_lists                 loudness-match + text front end + train/val split
                                            ─► data/3_training_lists/<name>  (speaker 0; a support voice is optional)
 STEP 3  src/3_train                       fine-tune the BASE model (models/base) for N steps  ─► checkpoints/<tag>
 STEP 4  src/4_demuffle   (optional)        lift clarity by a decoder weight edit (no retrain)  ─► checkpoints/<tag>_demuffled
 STEP 5  src/5_evaluate                     pronunciation score, Urdu error, kiosk listening page ─► results/
 STEP 6  src/6_speak + demo/                text ─► prepare_pk ─► Matcha ─► Vocos ─► 22,050 Hz audio
```

**The main rule:** list-building (step 2) and speaking (step 6) run text through the same function,
`pk_gazetteer.prepare_pk` (in `src/common/text/`). If a text rule changes, speaking changes at once; the training
lists keep the old text until step 2 is run again.

## Rebuild the shipped voice from scratch

**On GitHub, the raw and derived audio are not in the repo** (see `.gitignore`): the target voice's `wav/`, the
Rasa parquet shards, and the gain-matched `wavs/` folders under `data/3_training_lists/`. They're either not ours
to redistribute (Rasa, the base model) or too large for git. What **is** committed: the code, the training lists
as text (so you can see exactly what was trained on), the eval suites, and — the point of this repo — the trained
checkpoints (`checkpoints/voice/`, `checkpoints/voice_demuffled/`), so **using** the voice (speak, demo, evaluate)
works right after cloning plus `python src/fetch_pretrained.py` (below). Rebuilding from scratch additionally
needs the raw datasets re-fetched yourself: Rasa from [ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa)
shards 1–4, and the target voice's own audio from wherever you got (or build) that dataset.

Run everything from the `kiosk-tts` folder (the training lists store paths relative to it). Each step writes to a
**new** folder (`*_rebuild`, a new checkpoint tag), so the shipped copies are never overwritten. Steps 1, 3, 4 and 5
need an NVIDIA GPU (a 6 GB card is enough); step 2 is CPU.

```bash
cd ~/Desktop/kiosk-tts; PY=~/Desktop/kaani/envs/venv/bin/python

# STEP 1  gate the clips: keep the ones clearly its speaker with clean Urdu   -> data/2_clean/voice_clean_rebuild
$PY src/1_gate/gate.py

# STEP 2  training lists: the target clips (speaker 0) and the Rasa support (speaker 1)
#   --clean/--drop point at YOUR step-1 output; with no flags they read the shipped data/2_clean/voice_clean.
$PY src/2_build_lists/build_voice.py \
    --clean data/2_clean/voice_clean_rebuild --drop data/2_clean/voice_clean_rebuild/drop.txt \
    --out data/3_training_lists/voice_rebuild
$PY src/2_build_lists/build_rasa.py --shards 1,2,3,4   # -> data/3_training_lists/rasa_female_rebuild

# STEP 3  fine-tune the base model on YOUR rebuilt lists, 8,000 steps (~41 min)  -> checkpoints/voice_retrain
$PY src/3_train/train.py --tag voice_retrain \
    --data data/3_training_lists/voice_rebuild --support data/3_training_lists/rasa_female_rebuild/train.txt \
    --steps 8000 --ckpt-every 2000 --out-size 344 --bs 8 --accum 2 --lr 1e-4 \
    --target-weight 0.70 --spk-lr-mult 10 --seed-table v2 --no-row-b
# (or reproduce the shipped model on the shipped lists with the exact recipe:  bash src/3_train/train.sh voice_retrain)

# STEP 4  de-muffle: match its band shape, no retraining (~2 min)           -> checkpoints/<name>/model.ckpt
$PY src/4_demuffle/demuffle.py --name voice_retrain_demuffled --base checkpoints/voice_retrain/step=007999.ckpt

# STEP 5  evaluate
$PY src/5_evaluate/pron_eval.py --name voice_retrain --ckpt checkpoints/voice_retrain_demuffled/model.ckpt
$PY src/5_evaluate/pron_eval.py --compare voice_demuffled,voice_retrain      # paired comparison, no GPU
$PY src/5_evaluate/eval_target.py --name voice_retrain --ckpt checkpoints/voice_retrain_demuffled/model.ckpt
$PY src/5_evaluate/test_page.py  --name voice_retrain --ckpt checkpoints/voice_retrain_demuffled/model.ckpt

# STEP 6  speak with your retrain
$PY src/6_speak/speak.py --ckpt checkpoints/voice_retrain_demuffled/model.ckpt "..." -o out.wav
```

- `train.sh` is the exact recipe on the **shipped lists** (`data/3_training_lists/voice`, `rasa_female`) — use it
  to reproduce the shipped model. The `train.py` command above is the same recipe pointed at your step-2 lists.
- Don't want to re-gate or re-build? Skip steps 1–2 and run `bash src/3_train/train.sh <tag>` on the shipped lists.
  Want only to re-run the de-muffle on the shipped 8k model? Skip to step 4 with no `--base` (it reads
  `checkpoints/voice/step=007999.ckpt`).

### Why Rasa is here

The shipped fine-tune trains on **two** speakers: speaker 0 is the target voice (760 clips), speaker 1 is a
studio Urdu female voice from [ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa), kept at 30% of each
epoch. Rasa steadies the rare Urdu sounds (غ ق خ, aspirates) that a small YouTube-scraped dataset has few clean
examples of. The kiosk always speaks as speaker 0; **every English sound the model learns comes from the target
voice's own clips**, and no Indian-English data is used. That is why `data/1_source/rasa` and
`data/3_training_lists/rasa_female` are part of this project — the shipped voice cannot be rebuilt from scratch
without them.

**Rasa is optional for a voice of your own.** `train.py`'s `--support` flag is optional; leave it out to train on
your target voice alone (single-speaker, no speaker vectors at all). Whether Rasa helps a given new voice is
unproven — it was inherited with the recipe, never ablated. If your dataset already has clean, varied Urdu,
you likely don't need it; if it's thin on rare Urdu sounds, it may be worth trying both ways.

## Fine-tune on your own dataset

Every step below defaults to the shipped voice; give it your own audio and text instead and it fine-tunes a new
voice with the same recipe. All six steps are voice-agnostic — only step 1 depends on how your dataset is shaped.

### Step 1 — get your audio + text into the pipeline

If your dataset is **not** shaped like the shipped one (which is almost certainly the case), use
`import_voice.py`. It accepts:
- a folder of audio files, any common format (wav, flac, mp3, m4a, ogg, opus, aac, …), any sample rate;
- a transcript either as a `.csv` with a header (it looks for a file-name column and a text column by name, e.g.
  `clip_file`/`text`), or as plain lines of `<file name>|<transcript>`.

```bash
cd ~/Desktop/kiosk-tts; PY=~/Desktop/kaani/envs/venv/bin/python

$PY src/1_gate/import_voice.py --name myvoice --audio ~/data/myvoice/wav --text ~/data/myvoice/metadata.csv
```

It decodes every clip to 22,050 Hz mono (the whole pipeline assumes this rate — nothing later resamples, so this
step is not optional for a foreign-rate dataset), drops clips outside `--min-sec`/`--max-sec`, groups clips into
"recordings" by file-name prefix (so step 2 can hold one whole recording out for val/test, without any recording
leaking across the split), and writes `data/2_clean/myvoice/manifest.jsonl` — the same shape `gate.py` writes for
the shipped dataset. It ends by printing the exact step-2 command to run next, including which recording to hold
out.

Add `--gate` to also run the same TitaNet (is-it-one-voice) and Omnilingual (independent Urdu-text) checks the
shipped dataset was screened with — recommended for audio scraped from the wild, optional for studio-clean
recordings you already trust:

```bash
$PY src/1_gate/import_voice.py --name myvoice --audio ~/data/myvoice/wav --text ~/data/myvoice/metadata.csv --gate
```

If your dataset already has its own metadata format like the shipped one's (`train.txt`/`val.txt` +
`metadata.csv` with `reasr_similarity`/`ctc_similarity`/`promoted` columns), adapt `gate.py` instead of
`import_voice.py` — it is the version of step 1 written for that exact shape.

### Step 2 — build the training lists

Same script for any voice, `build_voice.py` — point it at what step 1 wrote:

```bash
$PY src/2_build_lists/build_voice.py --clean data/2_clean/myvoice --drop data/2_clean/myvoice/drop.txt \
    --held myvoice_heldout --out data/3_training_lists/myvoice
```

This matches the clips' loudness to the base model's own voice, runs the shared text front end
(`pk_gazetteer.prepare_pk`) so your text tokenises exactly like it will at speaking time, holds out the one
recording you name with `--held`, and writes `train.txt`/`val.txt`/`test.txt`.

**(Optional) a support voice.** If you want a second, support speaker (Rasa, or any other voice), build its list
the same way — `build_rasa.py` for Rasa specifically, or `build_voice.py` again pointed at a different clean
folder for any other support voice — and pass it to step 3's `--support`. Skip this and train single-speaker if
you don't want one; see ["Why Rasa is here"](#why-rasa-is-here).

### Step 3 — train

```bash
# single-speaker (your voice alone):
$PY src/3_train/train.py --tag myvoice \
    --data data/3_training_lists/myvoice \
    --steps 6000 --ckpt-every 2000 --out-size 344 --bs 8 --accum 2 --lr 1e-4 --seed-table v2 --no-row-b

# with a support voice (e.g. Rasa), the shipped recipe's flags:
$PY src/3_train/train.py --tag myvoice \
    --data data/3_training_lists/myvoice --support data/3_training_lists/rasa_female/train.txt \
    --steps 8000 --ckpt-every 2000 --out-size 344 --bs 8 --accum 2 --lr 1e-4 \
    --target-weight 0.70 --spk-lr-mult 10 --seed-table v2 --no-row-b
```

A few hundred clips (an hour or two) is enough to move the base model to a new voice; needs a CUDA GPU (a 6 GB
card is enough). `--steps`: the shipped voice used 8,000 with a support voice, 6,000 without; watch validation
loss and your ear, not just the step count. Output: `checkpoints/myvoice/step=*.ckpt`.

### Step 4 — de-muffle (optional)

Skip this if your fine-tune already sounds clear enough. If it comes out duller than the speaker's real
recordings (a common symptom of a small, scraped dataset), this closed-loop weight edit — no retraining, no
audio altered — can recover some of it:

```bash
$PY src/4_demuffle/demuffle.py --name myvoice_demuffled \
    --base checkpoints/myvoice/step=005999.ckpt --lists data/3_training_lists/myvoice
```

It fits a per-band bias on the decoder's last layer against some of your own training clips, and tests on your
held-out recording. Not every voice needs it — judge by ear.

### Step 5 — evaluate, and step 6 — speak

The evaluation and speaking scripts take `--ckpt` for any checkpoint, so they work unchanged on a new voice:

```bash
$PY src/5_evaluate/pron_eval.py --name myvoice --ckpt checkpoints/myvoice_demuffled/model.ckpt
$PY src/6_speak/speak.py --ckpt checkpoints/myvoice_demuffled/model.ckpt "..." -o out.wav
$PY demo/server.py --ckpt checkpoints/myvoice_demuffled/model.ckpt --port 8025   # a different port per voice
```

`pron_eval`'s "unseen word" flag and `make_open_suite.py`'s rare-word sampling read `data/2_clean/voice_clean` by
default (the shipped dataset); they skip that check quietly if it's absent, so they still run against a new
voice without it.

## Folder map

| Folder | What is in it |
|---|---|
| `samples/` | **Listen here first:** `samples/index.html` plays the shipped voice on kiosk answers, acronyms, a 90-sentence hard page, and its held-out video next to its real recordings. See [samples/README.md](samples/README.md). |
| `src/` | The code. Numbered folders are the steps; `common/` is shared code. See [src/README.md](src/README.md). |
| `demo/` | The type-and-listen web page (`server.py` + `index.html`) |
| `data/` | Data, numbered like the steps. See [data/README.md](data/README.md). The MARI knowledge base is `data/4_eval/knowledge_base/mari_energies_knowledge_base.md`. |
| `models/base/` | The base model fine-tuned from (`matcha_urdu.ckpt`) and its 350-symbol `vocab.json` |
| `models/vocoder/` | Vocos (`BSC-LT/vocos-mel-22khz`): turns the model's spectrogram into audio |
| `checkpoints/voice/` | The shipped voice after 8,000 steps (`step=007999.ckpt`), `args.json` (its recipe), and the exact rows it trained on (`train_ms.txt`, `val_ms.txt`) |
| `checkpoints/voice_demuffled/` | **The voice in use:** the 8k model plus the de-muffle edit (`model.ckpt`, `edit.json`, `val_ms.txt`) |
| `assets/` | Lookup files the text front end reads: English word lists, WordNet, NeMo number grammars, English↔Urdu spelling pairs |
| `vendor/Matcha-TTS/` | Matcha-TTS source code (MIT licence). It has our Urdu changes, so do not replace it with a fresh download. |
| `results/` | Evaluation outputs. The shipped voice's pronunciation scores are in `pron_voice_demuffled/`. |
| `logs/` | Training loss curves (`logs/voice/version_0/metrics.csv`) and the original run logs |
| `docs/` | `training_report.md`: the round that made the shipped voice, and why the recipe is what it is |

## Setup

**System programs:**
- `espeak-ng` 1.51 with the `ur`, `en-us` and `en-gb-x-rp` voices: English words become phone symbols through it.
- `ffmpeg`: listening pages and audio handling.
- For training and the gate, an NVIDIA GPU with CUDA (a 6 GB card is enough).

```bash
sudo apt install espeak-ng ffmpeg
```

**Python 3.12.** Until you make a new environment, the existing one works: `~/Desktop/kaani/envs/venv/bin/python`.
To make a new one:

```bash
python3.12 -m venv env
env/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
env/bin/pip install --no-deps diffusers==0.25.0 conformer==0.3.2
```

- `diffusers` must be exactly 0.25.0: Matcha uses parts that newer releases removed. `--no-deps` stops it from
  upgrading the other packages.
- `matcha_torch.py` patches the one incompatibility with newer `huggingface_hub` itself.
- `vendor/Matcha-TTS/matcha/utils/monotonic_align/` has a compiled extension built for Python 3.12 on x86-64. On
  another Python version, rebuild it from `vendor/Matcha-TTS` with:
  `python -c "from setuptools import Extension,setup; from Cython.Build import cythonize; import numpy,sys; sys.argv=['x','build_ext','--inplace']; setup(ext_modules=cythonize([Extension('matcha.utils.monotonic_align.core',['matcha/utils/monotonic_align/core.pyx'])]),include_dirs=[numpy.get_include()])"`

**Pretrained weights (not in the git repo — see `.gitignore`).** The base model and the vocoder are public on
Hugging Face; fetch them once:

```bash
python src/fetch_pretrained.py    # -> models/base/matcha_urdu.ckpt, models/vocoder/pytorch_model.bin
```

**Helper models (not copied, used from `~/.cache`).** The gate and the evaluation steps use these recognisers. On
this machine they are already in `~/.cache/huggingface` (and `~/.cache/torch/hub` for UTMOS); on a new machine they
download automatically the first time, if online.

| Model | Used by |
|---|---|
| `nvidia/speakerverification_en_titanet_large` | step 1 (is it one voice?) |
| `EmreAkgul/omniASR-CTC-300M-ONNX` (Omnilingual) | step 1 (independent Urdu check) |
| `Systran/faster-whisper-large-v3` | `eval_target` (Urdu character error rate) |
| `microsoft/wavlm-base-plus-sv` | `eval_target` (voice distance) |
| `facebook/wav2vec2-lv-60-espeak-cv-ft` | `pron_eval`, `phone_check` (phone recogniser) |

## Things to know

- **Every clip must be 22,050 Hz mono.** The base model, the mel spectrogram settings and the vocoder all assume
  it; nothing downstream resamples. `import_voice.py` resamples on the way in — if you build a clean folder by
  hand, resample first.
- **A retrain will not be bit-identical.** Training is not seeded; the same recipe run twice moves the
  knowledge-base score by about 1.5–3 points. Repeat a run before trusting a small difference, and let your ear decide.
- **`pron_eval --rebuild` gives a slightly different suite from the frozen one.** Compare checkpoints on the same
  suite; the frozen `data/4_eval/pron_suite.json` is the shared one.
- **`eval_target`'s voice score (`sim_shahed`) is a legacy distance to an old reference voice**, kept only to
  compare checkpoints — it is not "is this the shipped voice". For that, use the TitaNet check in
  `docs/training_report.md`.
- **Speaking settings:** 32 solver steps, temperature 0.35, speaker 0.
  - Long answers: split into sentences, speak each, join with 0.3 s of silence (`test_page.py` and `demo/server.py` do this).
  - The decoder was trained on 4-second windows, so sentence-at-a-time sounds best.
- **Known weak spot of the shipped voice:** it is duller than its real recordings, which are unusually bright. The
  de-muffle lifts the 2–4 kHz clarity partway but cannot add the loud peaks the model smooths away (see
  `docs/training_report.md`).

## The shipped voice's scores

From `results/` and `docs/training_report.md` (paired against an earlier voice on the same words and seeds):

| Measure | Shipped voice (`voice_demuffled`) | An earlier voice (same recipe) |
|---|---|---|
| Knowledge-base English words heard right (`pron_eval`) | 76.2% | 74.3% |
| Knowledge-base acronyms | 74.4% | 75.6% |
| General English words | 75.5% | 67.4% |
| Hard-page English (90 sentences) | 75.2% | 65.2% |
| Hard-page Urdu error | 0.151 | 0.130 |
| Tone kept vs its real recordings (choppiness proxy) | 100% | 87% |
| Voice is the target speaker (TitaNet, ≥ 0.8 is a match) | 1.01 | — |

- Knowledge-base words and general English are clearly better than the earlier voice; acronyms are within run noise.
- Pure Urdu (hard page group D) is 0.051–0.054.
- The voice is clearly the target speaker's, and the choppiness measures match its real recordings — the best of
  any model in the project so far.
