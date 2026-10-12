# kiosk-tts — Code & Pipeline Report

A guide to what every folder and file does, how data flows between them, and where to go to change a given behaviour.
For setup and the one-line commands, see [README.md](README.md). This report covers the code itself. (Line numbers
are close but may drift by a line or two after edits.)

---

## Contents

1. [What the project is](#1-what-the-project-is)
2. [The two pipelines at a glance](#2-the-two-pipelines-at-a-glance)
3. [Speaking: text → wav, call by call](#3-speaking-text--wav-call-by-call)
4. [The text front end in detail](#4-the-text-front-end-in-detail)
5. [The build pipeline, step by step](#5-the-build-pipeline-step-by-step)
6. [`src/common/`: shared modules, file by file](#6-srccommon-shared-modules-file-by-file)
7. [`demo/`, `vendor/`, and `src/paths.py`](#7-demo-vendor-and-srcpathspy)
8. [Non-code folders: data, models, checkpoints, assets, results, logs, docs](#8-non-code-folders)
9. [File formats you will meet](#9-file-formats-you-will-meet)
10. [Hidden wiring: monkey-patches and caches](#10-hidden-wiring-monkey-patches-and-caches)
11. ["I want to change X": where to go](#11-i-want-to-change-x-where-to-go)

---

## 1. What the project is

kiosk-tts fine-tunes Matcha-TTS into a Pakistani voice — originally the text-to-speech voice for the MARI
Energies kiosk, speaking the mixed Urdu/English text a RAG chatbot produces, but the pipeline is **voice-agnostic**:
every step works unchanged on a voice of your own. This report covers the shipped voice's recipe as a worked
example; see README.md for how to point the same steps at a new dataset.

- **Model:** Matcha-TTS (flow matching), fine-tuned from the Urdu base model `AhsanTalal/urdu-matcha-tts`.
- **Vocoder:** Vocos 22 kHz (`BSC-LT/vocos-mel-22khz`), which turns mel spectrograms into audio.
- **The checkpoint:** two speakers in one model: **0 = the target voice (the kiosk voice)**, 1 = Rasa (a studio Urdu
  support voice). Trained for 8,000 steps → `checkpoints/voice/step=007999.ckpt`, then a de-muffle weight edit →
  `checkpoints/voice_demuffled/model.ckpt`, **the checkpoint the kiosk speaks**.

**The key idea:** English words are not given to the model as letters. They become **Pakistani-English phone symbols**
built from the same symbol inventory Urdu uses, so the model only has to learn ~45 sounds. Urdu stays in Urdu script.
One function, `pk_gazetteer.prepare_pk`, produces this text for **both** training and speaking.

**Why there are two speakers.** Rasa never has to be spoken at the kiosk; it is support data (30% of each epoch) that
keeps rare Urdu sounds (غ ق خ, aspirates) alive while fine-tuning on a small, scraped voice. Every English sound the
model learns comes from the target voice's own clips, and no Indian-English data is used. Rasa (or any support
voice) is optional for a new voice — see README.md.

---

## 2. The two pipelines at a glance

```
 ┌────────────────────────── BUILD PIPELINE (offline, GPU except step 2) ─────────────────────┐
 │                                                                                             │
 │ A speech dataset (built outside this repo, or imported with src/1_gate/import_voice.py)    │
 │       ──► data/1_source/voice                                                              │
 │                                                                                             │
 │ STEP 1  src/1_gate/gate.py              keep clips clearly one voice, clean Urdu            │
 │             data/1_source/voice ──► data/2_clean/voice_clean                            │
 │ STEP 2  src/2_build_lists/                                                                  │
 │             build_voice.py (spk 0)    ──► data/3_training_lists/voice                    │
 │             build_rasa.py    (spk 1)    ──► data/3_training_lists/rasa_female                │
 │ STEP 3  src/3_train/train.py (+ train.sh)                                                   │
 │             models/base + lists ──► checkpoints/<tag>/step=*.ckpt, logs/<tag>/              │
 │ STEP 4  src/4_demuffle/demuffle.py      ──► checkpoints/<tag>_demuffled/model.ckpt          │
 │ STEP 5  src/5_evaluate/  pron_eval / eval_target / test_page ──► results/                   │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────── SPEAK PIPELINE (runtime, CPU or GPU) ──────────────────────┐
 │ text ─► pk_gazetteer.prepare_pk ─► tts.clean ─► tts.encode ─► Matcha.synthesise    │
 │      (src/common/text/)                          (32 ODE steps, temp 0.35, spk 0)  │
 │      ─► mel ─► Vocos.decode ─► tts.normalize ─► 22,050 Hz wav                      │
 │ Entry points: src/6_speak/speak.py (one wav), demo/server.py (web page),           │
 │               5_evaluate/* (render for scoring)                                    │
 └────────────────────────────────────────────────────────────────────────────────────┘
```

**Folder numbering rule:** `src/N_*` feeds `data/N_*`. Every rebuild writes to a *new* folder (`*_rebuild`, a new
checkpoint tag), so the shipped copies are never overwritten.

---

## 3. Speaking: text → wav, call by call

This is the path taken by [src/6_speak/speak.py](src/6_speak/speak.py), [demo/server.py](demo/server.py), and
every evaluation script.

| # | Call | File | What happens |
|---|---|---|---|
| 1 | `import paths` | [src/paths.py](src/paths.py) | Puts every `src/` subfolder on `sys.path`; defines all file locations. |
| 2 | Pick checkpoint | [speak.py:44](src/6_speak/speak.py#L44) | Default `paths.VOICE` (`checkpoints/voice_demuffled`), or any `--ckpt`. |
| 3 | `spk_cond.load_for_inference(ckpt, dev, spk=0)` | [spk_cond.py:100](src/common/model/spk_cond.py#L100) | Builds the model (via `matcha_torch._load`), attaches the speaker embeddings, loads their weights, and sets default speaker 0. |
| 3a | ↳ `matcha_torch._load` | [matcha_torch.py:95](src/common/model/matcha_torch.py#L95) | One-slot cache: vocab, model, and vocoder. |
| 3b | ↳ `matcha_torch.load_model` | [matcha_torch.py:63](src/common/model/matcha_torch.py#L63) | `MatchaTTS(**hparams)`; fixes `n_vocab` to 350; restores `mel_mean`/`mel_std` (otherwise the output is garbage). |
| 3c | ↳ `vocoder._load_vocos_22k` | [vocoder.py:15](src/common/model/vocoder.py#L15) | Vocos backbone and ISTFT head from `models/vocoder/`. |
| 4 | `pk_gazetteer.prepare_pk(text)` | [pk_gazetteer.py:231](src/common/text/pk_gazetteer.py#L231) | **The text front end.** Covered in [section 4](#4-the-text-front-end-in-detail). |
| 5 | `tts.clean(s)` | [tts.py:73](src/common/text/tts.py#L73) | NFC; any character not in the 350-symbol vocab becomes a space; whitespace collapsed. |
| 6 | `tts.encode(s)` | [tts.py:80](src/common/text/tts.py#L80) | Chars → vocab ids, with blank `0` between every symbol (Matcha's `add_blank`). |
| 7 | `model.synthesise(x, xl, n_timesteps=32, temperature=0.35, length_scale)` | vendor + patched by [spk_cond.py:49](src/common/model/spk_cond.py#L49) | Encoder → durations → flow-matching decoder → mel. Speaker vectors are added to `mu` and the time embedding. |
| 8 | `vocos.decode(mel)` | [vocoder.py](src/common/model/vocoder.py) | Mel → waveform. |
| 9 | `tts.normalize(w)` | [tts.py:90](src/common/text/tts.py#L90) | RMS 0.06, peak ≤ 0.97. |
| 10 | `sf.write(out, w, 22050)` | speak.py | Writes the wav and prints the exact symbols the model read. |

**Default speaking settings:** 32 solver steps, temperature 0.35, seed 100 in `speak.py`, speaker 0. Long text is split
into sentences, each is spoken separately, and the results are joined with 0.3 s of silence. That happens in
`demo/server.py` and `test_page.py`, not in `speak.py`. The decoder was trained on 4 s windows, so sentence-at-a-time
sounds best.

---

## 4. The text front end in detail

Everything lives in [src/common/text/](src/common/text/). The single entry point is:

```python
pk_gazetteer.prepare_pk(text, letters="ur")
    = frontend.prepare( gazetteer.apply( text_norm.normalize(text) ), letters="ur" )
```

### 4.1 Stage A: `text_norm.normalize`, cleaning what an LLM writes

[text_norm.py:325](src/common/text/text_norm.py#L325) runs these passes **in this order**:

| Pass | Example |
|---|---|
| markup | markdown, bullets, links, emoji → plain text |
| greeting | `Assalam-o-Alaikum` / `AOA` / `السلام علیکم` → `اَسّلامُ علیکم` |
| web | `info@mari.com` → "info at mari dot com" |
| dates | `23 March 2025` → words |
| standards | ISO-style codes |
| money | `PKR 5 bn`, `US$` → amount + rupees/dollars (روپے/ڈالر in Urdu) |
| units | `km`, `MW`, `°C` after a number → words |
| decimals | `39.7` |
| phone parens | `(+92) 51` → `+92 51` |
| identifiers | long digit strings and hyphenated codes → digit by digit |
| urdu times | `10:30` in Urdu → دس بج کر تیس منٹ |
| ranks & codes | `Brig (Retd)`, `Lt Gen` |
| mixed numbers | a number next to an Urdu scale word (لاکھ, کروڑ…) |
| acronyms | plurals and possessives (`CEOs`), `E&P`, dotted `A.F.`, 7–8-letter capitals |
| camel case | `MariEnergies` → `Mari Energies` |
| abbreviations | words without a trailing period |

Other things in this file:
- `PK_PHONES` ([L46](src/common/text/text_norm.py#L46)): exact phone strings for specific names.
- `EXTRA_WORD_ACRONYMS` ([L39](src/common/text/text_norm.py#L39)): acronyms said as words.
- `frontend()` ([L58](src/common/text/text_norm.py#L58)) **patches the `frontend` module** with these lists (see [section 10](#10-hidden-wiring-monkey-patches-and-caches)).
- `python src/common/text/text_norm.py --test` runs the tests.

### 4.2 Stage B: `pk_gazetteer.apply`, Pakistani names to Urdu script

[pk_gazetteer.py:121](src/common/text/pk_gazetteer.py#L121):
- `PHRASES` (multi-word, matched first) and `WORDS` map Latin spellings of Pakistani cities, provinces, people,
  companies and Islamic terms to Urdu script. Examples: `Quetta` → کوئٹہ, `Mari Energies` → ماڑی انرجیز.
- Longest match wins, and matching is case-insensitive.
- Words that are also ordinary English (Jazz, Lucky, Fast) are deliberately left out.

### 4.3 Stage C: `frontend.prepare`, splitting Urdu and English and producing symbols

[frontend.py:364](src/common/text/frontend.py#L364), in order:

1. `drop_parenthetical_gloss`: `ایم ایم بی او ای (MMBOE)` is no longer read twice.
2. `expand_abbreviations`: `Dr.`, `Mr.`, and similar.
3. `roman_urdu_guard`: Roman-Urdu words → Urdu script (`text_ipa.ROMAN_URDU` + `ROMAN_URDU_EXTRA`).
4. `expand_codes`: `2P`, `FY2024-25`, `COP29`; the digits become **English** number words.
5. `english_numbers` ([L216](src/common/text/frontend.py#L216)): numbers in English context use NeMo grammars (`assets/nemo_cache`).
6. `urdu_numbers.expand_numbers`: the remaining numbers become Urdu words (lakh/crore scale).
7. Every Latin span goes to `_span_pieces` ([L319](src/common/text/frontend.py#L319)), which decides each token's fate:

   | Token looks like | Becomes |
   |---|---|
   | in `PHONE_OVERRIDE` ([L69](src/common/text/frontend.py#L69), plus `PK_PHONES`) | literal phone string |
   | Roman numeral after a Capitalised word (`World War II`) | English number word |
   | all-caps ≤ 6, not in `ACRONYM_WORDS`, not a dictionary heading | **spelled** letter by letter; with `letters="ur"` the Urdu letter names (`ETF` → ای ٹی ایف) |
   | plural acronym (`CEOs`) | spelled + `z` |
   | mixed-case abbreviation (`PhD`) or vowel-less (`kWh`) | spelled |
   | other all-caps (`NADRA`) | English word |
   | anything else | `names.route(tok)`, see below |

8. `names.route` ([names.py:201](src/common/text/names.py#L201)) decides English or name:
   1. English dictionary word, inflection, or compound → `("en", tok)`.
   2. In the transliteration lexicon (`assets/translit_pairs.json`) → `("ur", urdu_script)`.
   3. Otherwise a character-trigram classifier: a Roman-Urdu-looking name → `("ph", roman_phones(tok))`; else English.
9. Each piece is rendered:
   - `en` → `phones.english_span(...)`, **replaced at runtime** by `pk_gazetteer._english_span_pk` ([L195](src/common/text/pk_gazetteer.py#L195)):
     `en_g2p.shape` → espeak-ng twice (`en-us`, `en-gb-x-rp`) → `_gb_word` merges (British vowels, rhotic r put back) →
     `phones.EN_MAP` to Pakistani-English phones. The rule set is `PK_EN_RULES` (default `B`).
   - `ph` → passed through as is. `ur` → `urdu_text_norm.normalize_for_tts(latin="drop")`.
10. `tts.clean`: only vocab symbols survive.

**Result example** (from a training list):
`جو ʈˈɔp lˈaːɪn اچھے ہی rɪʈˈərnz دے رہے ہیں` (Urdu script with Pakistani-English phones inline).

**Handy commands:**
```bash
python src/common/text/pk_gazetteer.py --prepare "Lt Gen Anwar Ali Hyder visited G-10/4 on 23 March 2025"
python src/common/text/pk_gazetteer.py --test
python src/common/text/text_norm.py --test
python src/common/text/names.py Faheem Waziristan Huawei
```

---

## 5. The build pipeline, step by step

### STEP 1: [src/1_gate/gate.py](src/1_gate/gate.py), keep her clean clips

- **In:** `data/1_source/voice/{train.txt, val.txt, metadata.csv, wav/}` (864 clips, 1.74 h; already de-musicked and
  cut by `~/Desktop/urdu-dataset-builder`).
- **Out:** `<out>/{manifest.jsonl, drop.txt, keep_summary.json, urdu_check.jsonl, titanet_centroid.npy}` and a `wav`
  symlink. The shipped copy is `data/2_clean/voice_clean` (782 of 864 kept).
- **Drop rule** — a clip is dropped for any of:
  - `reasr_C`: the transcriber's Whisper re-decode agreed < 0.70 with the transcript;
  - `promoted`: an auto-released review clip (the dataset README's "strict" option);
  - `loop` ([is_loop L54](src/1_gate/gate.py#L54)): one word is > 40% of a 6+ word text;
  - `voice`: TitaNet similarity < `VOICE_MIN` 0.70 ([L47](src/1_gate/gate.py#L47)) to her centroid (300 random clips outside the held-out video);
  - `urdu_cer`: an independent Omnilingual ASR CER above the p90 of a known-clean line, `ANAM_URDU_CER_LINE` ([L51](src/1_gate/gate.py#L51), a fixed number; no Anam audio is read).
- The Urdu check is resumable through `urdu_check.jsonl`. Helpers are in [gate_utils.py](src/1_gate/gate_utils.py)
  (`Omni` Omnilingual CTC ONNX on CPU, `urdu_cers` semi-global Urdu-only scoring, `Embedder` TitaNet via NeMo).
- Models from the HF cache: `nvidia/speakerverification_en_titanet_large`, `EmreAkgul/omniASR-CTC-300M-ONNX`.

### STEP 2a: [src/2_build_lists/build_voice.py](src/2_build_lists/build_voice.py), her lists (speaker 0)

- **In:** `data/2_clean/voice_clean` (`manifest.jsonl` + `wav/` + `drop.txt`) and `data/1_source/anchor/*.wav` (the
  base voice's loudness). Defaults are already pointed at these.
- **Out:** `<out>/{wavs/, train.txt, val.txt, test.txt, test_raw.json, summary.json}`. Shipped: `data/3_training_lists/voice`.
- **What it does:**
  1. **Split:** holds out one whole video — `voice_-gI_ALIlkR8` (the dataset's own validation video), or `--held`.
     From it: 12 test clips and up to 10 val clips.
  2. **Gain:** solves one global gain so the clips' mean log-mel matches the anchor voice. Each clip is capped so its
     peak stays under `PEAK` 0.84 ([L45](src/2_build_lists/build_voice.py#L45)), so `soft_limit` never alters a training clip.
  3. **Text:** Row A = `pk_gazetteer.prepare_pk(text)`; Row B = row A with some Urdu words replaced by espeak-ur phones
     (`dataset_utils.rows_from`, p=0.5).
  4. **Tokenizer check:** Matcha's `text_to_sequence` must equal `tts.encode`, or the clip is dropped as `tokenizer`.
  5. Writes row A then row B per clip, **unshuffled**. Training's `--no-row-b` keeps every other line (row A).
- Options: `--clean`, `--extra`, `--drop`, `--held`, `--out`; `--alter` must be `none`. The same script builds any
  target speaker's lists — point the flags elsewhere.

### STEP 2b: [src/2_build_lists/build_rasa.py](src/2_build_lists/build_rasa.py), the Rasa support list (speaker 1)

- **In:** `data/1_source/rasa/*.parquet` (AI4Bharat Rasa Urdu, shards 1–4).
- **Out:** `<out>/{wavs/, train.txt, summary.json}`. Shipped: `data/3_training_lists/rasa_female` (893 clips, 1.51 h).
- Keeps female speakers, neutral styles (WIKI/CONV/BOOK/NEWS) and clips ≤ 12 s. Same front end and row A/B logic, own
  loudness match. Lines are **shuffled**. Purpose: keeps rare Urdu sounds alive during fine-tuning.

### STEP 3: [src/3_train/](src/3_train/), fine-tune

**[train.py](src/3_train/train.py)**, `main` at [L106](src/3_train/train.py#L106):

1. Loads **always the untouched base checkpoint** (`models/base/matcha_urdu.ckpt`) with `strict=True`, and sets
   `out_size=344` (4 s decoder windows).
2. **Embedding init:** each English phone symbol's embedding row starts from the Urdu letter with the same sound
   (`phone_init.INIT_FROM`, overlaid with `SEED_V2` [L39](src/3_train/train.py#L39) when `--seed-table v2`, which gives
   contrasting vowels different starts).
3. **Speakers:** with `--support`, `spk_cond.attach(model, 2)`. `tag_rows` ([L186](src/3_train/train.py#L186)) rewrites
   lines as `wav|spk|text`; `--no-row-b` keeps every other line; `--target-weight 0.70` repeats/trims Rasa rows so the
   target voice makes up 70% of each epoch. Writes `train_ms.txt` and `val_ms.txt`.
4. **Optimiser:** Adam at lr 1e-4. Speaker parameters run at `--spk-lr-mult` × lr (10×).
5. **Data:** Matcha's `TextMelDataModule` (cleaner `urdu_cleaners_graphemes`, `add_blank`, 22,050 Hz, 80 mels, the base
   checkpoint's `data_statistics`).
6. **Trainer:** Lightning, fp16-mixed, grad-accum 2, clip 5.0. Saves every 2,000 steps to `checkpoints/<tag>/step=*.ckpt`,
   logs to `logs/<tag>/version_0/metrics.csv`, writes `args.json`.
7. Unused experiment flags remain: `--support2`, `--noise-channel`, `--mask-pauses` (`pause_mask` [L56](src/3_train/train.py#L56)).

**Other files:**
- **[train.sh](src/3_train/train.sh):** the exact recipe, every flag commented: 8,000 steps, bs 8, accum 2,
  target-weight 0.70, spk-lr-mult 10, seed-table v2, no-row-b. ~41 min on a 6 GB GPU.
- **[phone_init.py](src/3_train/phone_init.py):** the `INIT_FROM` dict (phone symbol → Urdu letter(s)).

### STEP 4: [src/4_demuffle/](src/4_demuffle/), lift the target voice's 2–4 kHz clarity (weights only, optional)

A trained voice can come out duller than its real recordings (the shipped voice: clarity 0.93 vs 2.07). This adds a
per-band bias to the decoder's last layer, fitted in closed loop — no retraining, no audio altered. Skip it if a
voice already sounds clear enough.

**[demuffle.py](src/4_demuffle/demuffle.py)** ([main L42](src/4_demuffle/demuffle.py#L42)) supplies the target
voice's own reference clips (via `--lists`) and drives the shared engine:
- fit: `--fit-clips` of its own training clips (`<lists>/train.txt`, `random.Random(0)`);
- test: its held-out recording (val + test), scored once at the end;
- no speaker swap — speaker 0 is already the target voice.

**[voice_edit.py](src/4_demuffle/voice_edit.py)** ([main L119](src/4_demuffle/voice_edit.py#L119)) is the engine:
- **`--demuffle s`:** closed-loop fit of an 80-band bias on `decoder.estimator.final_proj.bias`: render the references,
  measure the band-shape gap to the real recordings (`spectra`, `curve_from`), move the bias by `STEP` 0.5 × the error,
  capped at ±6 dB (`CAP`), repeat `ITERS` 5 times ([L48–49](src/4_demuffle/voice_edit.py#L48)). `s = 1.0` removes the whole gap.
- **`--swap` / `--blend` / `--pace`** exist (kept from an earlier project that used them to swap in a studio support
  voice) but demuffle.py does not use them.
- **Out:** `checkpoints/<name>/{model.ckpt, edit.json, val_ms.txt}`.

### STEP 5: [src/5_evaluate/](src/5_evaluate/), measure

| File | Role | In → Out |
|---|---|---|
| [pron_eval.py](src/5_evaluate/pron_eval.py) | **Main accuracy score** (details below). | `data/4_eval/pron_suite.json` (frozen, 3,277 renders) → `results/pron_<name>/{scores.json, words.jsonl, index.html, audio/}` |
| [phone_check.py](src/5_evaluate/phone_check.py) | Library: `tokens`, `canon` (equivalence classes such as t/ʈ), `align` (best-window edit distance). Run alone, it checks that the training labels' phones match what she says. | `data/2_clean/voice_clean` → `results/phone_check/report.json` |
| [make_open_suite.py](src/5_evaluate/make_open_suite.py) | Word lists (`GEN_EN`, `GEN_CS`, `MARI_EN`, `MARI_CS`, `ACR_GEN`) and `kb_acronyms()`. | knowledge base → `data/4_eval/open_suite_rebuild.json` |
| [eval_target.py](src/5_evaluate/eval_target.py) | Whisper Urdu CER on `urdu_parity` + `urdu_rare`; English word hit rate; WavLM distance to the legacy **Shahed** reference and to Rasa | `data/4_eval/eval_suite.json` → `results/eval_<name>/{index.html, scores.json}` |
| [test_page.py](src/5_evaluate/test_page.py) | Kiosk listening page: 30 KB answers (split into sentences, 0.3 s gaps) and 143 names (`"جی، <name>۔"`), with right/wrong buttons that export JSON | `kb_test_sentences.txt`, `kb_test_names.txt` → `results/<name>/index.html` + mp3s |

**How `pron_eval.py` works** ([build_suite L123](src/5_evaluate/pron_eval.py#L123), [main L373](src/5_evaluate/pron_eval.py#L373)):
1. Renders each item with seed = item index, so two checkpoints compare on identical noise.
2. Uses the model's own durations to find where each target word sits, decodes only that stretch ±80 ms with
   wav2vec2-espeak phone recognition, and aligns the expected phones. A word passes at distance `d ≤ PASS_D` 0.25 ([L63](src/5_evaluate/pron_eval.py#L63)).
3. Minimal pairs (`PAIRS` [L72](src/5_evaluate/pron_eval.py#L72)) pass when the word's own phones fit better than its rival's.
4. Groups: `PRIMARY` = kb_word/kb_acr/kb_sent ([L69](src/5_evaluate/pron_eval.py#L69)); `GUARD` = pair/gen_word/gen_acr.
5. `--rebuild` regenerates the suite to `pron_suite_rebuild.json`; `--compare a,b` gives a paired bootstrap difference, no GPU.

### STEP 6: [src/6_speak/speak.py](src/6_speak/speak.py), speak one text

See [section 3](#3-speaking-text--wav-call-by-call). Flags: `--ckpt`, `--seed`, `--steps`, `--temp`, `--length-scale`
(if ≠ 1 it calls `model.synthesise` directly instead of `tts.synth`).

---

## 6. `src/common/`: shared modules, file by file

### `common/text/`: the text front end

| File | What it contains | Key symbols |
|---|---|---|
| [pk_gazetteer.py](src/common/text/pk_gazetteer.py) | **Entry point.** Pakistani name table and the English phone rules (British-based). | `prepare_pk` L231, `apply` L121, `_english_span_pk` L195, `_gb_word` L167, `EN_RULES` L158 |
| [text_norm.py](src/common/text/text_norm.py) | LLM text cleanup (section 4.1); patches `frontend`. | `normalize` L325, `frontend()` L58, `PK_PHONES` L46, `EXTRA_WORD_ACRONYMS` L39 |
| [frontend.py](src/common/text/frontend.py) | Urdu/English split, acronym spelling, codes, numbers, Roman-Urdu guard. | `prepare` L364, `_span_pieces` L319, `PHONE_OVERRIDE` L69, `english_numbers` L216 |
| [phones.py](src/common/text/phones.py) | The shared phone inventory and the espeak → phone maps. | `PHONES` L36, `EN_MAP` L59, `espeak` L90, `english_span` L169 (replaced at runtime) |
| [names.py](src/common/text/names.py) | English-or-name router. | `route` L201 |
| [en_g2p.py](src/common/text/en_g2p.py) | Spelling shaping before espeak; an in-process ctypes espeak `ipa()`. | `shape`, `ipa`, `OVERRIDE` |
| [text_ipa.py](src/common/text/text_ipa.py) | Legacy module; **only its `ROMAN_URDU` table is used**, loaded by file path. | `ROMAN_URDU` |
| [urdu_text_norm.py](src/common/text/urdu_text_norm.py) | Canonical Urdu normalisation; makes sure only *trained* symbols reach the model. | `normalize_for_tts` |
| [urdu_numbers.py](src/common/text/urdu_numbers.py) | Urdu number words (lakh/crore scale). | `expand_numbers` |
| [tts.py](src/common/text/tts.py) | Vocab and low-level synthesis helpers. `prepare(mode)` is a legacy path; only `"script"` is still used, by eval_target for the base model. | `vocab` L37, `clean` L73, `encode` L80, `normalize` L90, `synth` L97 |

### `common/model/`: the model

| File | What it contains |
|---|---|
| [matcha_torch.py](src/common/model/matcha_torch.py) | `_prepare_imports` (L37) adds `vendor/Matcha-TTS` to the path, stubs `gdown`, aliases `huggingface_hub.cached_download`. `load_model` (L63) builds MatchaTTS from a checkpoint. `_load` (L95) is a **one-slot cache** of `{stoi, model, vocos}`. |
| [spk_cond.py](src/common/model/spk_cond.py) | `SpkCond` (L21) holds two zero-initialised embeddings: `emb_t` (n_spk × 1024, added to the decoder time embedding) and `emb_mu` (n_spk × 80, added to the encoder prior). The text encoder and duration predictor get **no** speaker input, so pronunciation and rhythm are shared. `attach` (L49) monkey-patches the forward methods. `has_spk_cond` (L95). `load_for_inference` (L100). |
| [vocoder.py](src/common/model/vocoder.py) | `_load_vocos_22k` (L15): Vocos backbone and ISTFT head only, skipping the mel feature extractor. |

### `common/dataset/`

| File | What it contains |
|---|---|
| [dataset_utils.py](src/common/dataset/dataset_utils.py) | `rows_from` (row B builder), `mel_mean` (loudness), `soft_limit` (knee 0.85, ceiling 0.98), `_TOKEN` regex. Used by the step-2 list builders. |

---

## 7. `demo/`, `vendor/`, and `src/paths.py`

### [src/paths.py](src/paths.py)
- Every script starts with `sys.path.insert(0, <src>)` then `import paths`, which puts `common/{text,model,dataset}`
  and the numbered step folders on `sys.path`, so modules import each other by bare name (`import pk_gazetteer`).
- Defines every location: `BASE_CKPT`, `BASE_VOCAB`, `VOCODER`, `VOICE_8K`, `VOICE`, `ASSETS`, `TRANSLIT`,
  `NEMO_CACHE`, the `DATA` subfolders (`VOICE_SOURCE`, `RASA_SOURCE`, `ANCHOR`, `VOICE_CLEAN`, `VOICE_LISTS`,
  `RASA_LISTS`, `EVAL`), `RESULTS`, `LOGS`, `OUTPUT`.
- **To move a file, update it here.**

### [demo/](demo/)
- **[server.py](demo/server.py):** standard-library HTTP server, no framework.
  - `Voice` ([L59](demo/server.py#L59)) loads the model once and warms up NeMo and espeak.
  - `speak` ([L75](demo/server.py#L75)) splits sentences (`sentences` [L46](demo/server.py#L46), `ABBR_END` [L41](demo/server.py#L41)
    avoids splitting after "Dr.", "Lt. Gen."), then runs `prepare_pk` → `clean` → `encode` → `synthesise` (32 steps,
    temp 0.35, `length_scale = 1/speed`) → Vocos → `normalize`, joining sentences with 0.3 s gaps. A lock allows one render at a time.
  - Endpoints: `GET /` (page), `GET /api/info` (checkpoint), `POST /api/speak` `{text, speed 0.6–1.5, seed}`.
  - Default port: 8024. `--ckpt` serves any checkpoint.
- **[index.html](demo/index.html):** the page: a text box, speed control, playback, and history; shows what the model "read".

### [vendor/Matcha-TTS/](vendor/Matcha-TTS/)
Matcha-TTS source, not pip-installed. It contains local Urdu changes, so **do not replace it with a fresh copy**:
- `matcha/text/symbols.py`: generated from the base model's 350-symbol `vocab.json`.
- `matcha/text/cleaners.py`: `urdu_cleaners_graphemes` — NFC, out-of-vocab → spaces, whitespace collapse. Must match `tts.clean`.
- `matcha/utils/monotonic_align/`: a prebuilt Cython extension for Python 3.12 x86-64.
- Model code: `models/matcha_tts.py`, `models/components/*`, `data/text_mel_datamodule.py`.

---

## 8. Non-code folders

| Folder | Size | Contents |
|---|---|---|
| [data/1_source/](data/1_source/) | ~1.4 GB | Raw inputs, read-only: `voice/` (864 clips plus metadata), `rasa/` (4 parquet shards, the support voice), `anchor/` (50 base-voice clips, the loudness target) |
| [data/2_clean/](data/2_clean/) | | `voice_clean/` (the gate output: `manifest.jsonl`, `drop.txt`, `keep_summary.json`, `urdu_check.jsonl`, `titanet_centroid.npy`, and a `wav` symlink) |
| [data/3_training_lists/](data/3_training_lists/) | | `voice/` (exact speaker-0 lists + gain-matched wavs), `rasa_female/` (exact speaker-1 list) |
| [data/4_eval/](data/4_eval/) | | Frozen test sets: `pron_suite.json`, `open_suite.json`, `eval_suite.json`, `kb_test_*.txt`, `shahed_ref/`, and **`knowledge_base/mari_energies_knowledge_base.md`** (the chatbot's source text) |
| [models/base/](models/base/) | 261 MB incl. vocoder | `matcha_urdu.ckpt` (base model) and `vocab.json` (350 symbols) |
| [models/vocoder/](models/vocoder/) | | Vocos `config.yaml` and `pytorch_model.bin` |
| [checkpoints/voice/](checkpoints/voice/) | 70 MB | `step=007999.ckpt` (the 8k model), `args.json` (its recipe), `train_ms.txt`, `val_ms.txt` |
| [checkpoints/voice_demuffled/](checkpoints/voice_demuffled/) | 70 MB | **The voice in use:** `model.ckpt`, `edit.json` (bias curve, fit/test gaps), `val_ms.txt` |
| [assets/](assets/) | 23 MB | `dict/` (English word lists), `nltk_data/` (WordNet), `nemo_cache/` (compiled NeMo number grammars), `translit_pairs.json`, `extra_english.txt` |
| [results/](results/) | | `pron_voice_demuffled/` (the shipped voice's scores), `sample_voice.wav` |
| [logs/](logs/) | | `voice/version_0/metrics.csv` (loss curve), `hparams.yaml`, the step logs under `voice/`, and `data_build/` |
| [docs/](docs/) | | [docs/README.md](docs/README.md) and `pk-r16_REPORT.md`: the round that made this voice (old kaani paths inside) |

The helper recogniser models are **not** in the repo; they load from `~/.cache/huggingface`:

| Model | Used by |
|---|---|
| TitaNet, Omnilingual | step 1 (the gate) |
| faster-whisper large-v3 | `eval_target` |
| WavLM-sv | `eval_target` |
| wav2vec2-espeak | `pron_eval`, `phone_check` |

---

## 9. File formats you will meet

| File | Format |
|---|---|
| `data/3_training_lists/*/train.txt` | `<wav path>\|<text the model reads>`. Two lines per clip: row A, then row B. |
| `checkpoints/<tag>/train_ms.txt` | `<wav path>\|<speaker id>\|<text>`, the rows training actually used, shuffled. |
| `data/2_clean/voice_clean/manifest.jsonl` | `{clip, video, seg, text, queue, flags, duration, words, english_words, reasr, ctc, promoted, spk_sim}` |
| `data/2_clean/voice_clean/urdu_check.jsonl` | `{clip, omni (Omnilingual transcript), cer}` |
| `data/4_eval/pron_suite.json` | A list of `{id, group, text, targets:[{word, exp, unseen}], rival?}` |
| `results/pron_*/scores.json` | Per group: `{words, pass_rate, mean_d, not_located, unseen_pass_rate, pair_accuracy?}` |
| `results/pron_*/words.jsonl` | One row per scored word: `{id, group, word, d, exp, heard, t0, t1, rival_d?, pair_ok?}` |
| `checkpoints/*/edit.json` | `{base, swap, demuffle, bias_db[80], fit_err_db_by_iter, gap_db_test_before/after, …}` |
| `kb_test_sentences.txt` / `kb_test_names.txt` | `category\|question\|answer` / `group\|name`; lines starting with `#` are comments. |

---

## 10. Hidden wiring: monkey-patches and caches

If something seems to "come from nowhere", check here:

1. **`pk_gazetteer._phones()` replaces `phones.english_span`** with `_english_span_pk`. Any code that calls
   `phones.english_span` after `prepare_pk` has run gets the British/Pakistani rules, not the original en-us ones.
   That is deliberate, so training and speaking agree.
2. **`text_norm.frontend()` mutates the `frontend` module** ([L58](src/common/text/text_norm.py#L58)): removes
   SECP/HEC/KSE from `ACRONYM_WORDS`, adds `EXTRA_WORD_ACRONYMS`, merges `PK_PHONES` into `PHONE_OVERRIDE`, swaps
   `_MIXED_ABBR`, replaces `roman_urdu_guard`. Editing the lists in `frontend.py` alone may be overridden.
3. **`spk_cond.attach` patches** `model.encoder.forward`, `model.decoder.estimator.time_mlp.forward`, `model.forward`
   and `model.synthesise`. The current speaker is held in `model._cur_spk`.
4. **`matcha_torch._load` is a one-slot cache** keyed by `(ckpt, device)`. Loading a second checkpoint evicts the first.
   `voice_edit.model_of_fresh` clears it to force a reload after rewriting a checkpoint on disk.
5. **`frontend._roman_urdu` loads `text_ipa.py` by file path** (importlib), so `text_ipa.py` looks unused but isn't.
6. **`PK_EN_RULES` env var** (default `B`) switches the English phone rules. If it differs between training and
   speaking, the labels and the speech no longer match.
7. **`--no-row-b` takes `lines[::2]`.** That is correct for the target voice's unshuffled lists (keeps row A). For
   Rasa's shuffled list it keeps a random half of A and B rows — which is what the recipe trains on.

---

## 11. "I want to change X": where to go

| I want to… | Go to |
|---|---|
| Fix how a **Pakistani name / city / company** is said | `WORDS` / `PHRASES` in [pk_gazetteer.py](src/common/text/pk_gazetteer.py); for an exact phone string, `PK_PHONES` in [text_norm.py](src/common/text/text_norm.py#L46) |
| Fix how an **English word** is pronounced | `en_g2p.OVERRIDE` (respelling before espeak) or `PHONE_OVERRIDE` in [frontend.py](src/common/text/frontend.py#L69) (literal phones) |
| Change a **Pakistani-English sound rule** | `EN_MAP` in [phones.py](src/common/text/phones.py#L59); British-vowel logic in `_gb_word` / `_english_span_pk` in [pk_gazetteer.py](src/common/text/pk_gazetteer.py#L167) |
| Make an acronym be **said as a word** instead of spelled | `EXTRA_WORD_ACRONYMS` in [text_norm.py](src/common/text/text_norm.py#L39) |
| Change **how letters are named** (A → اے, Z → زیڈ) | `LETTER_UR` in [frontend.py](src/common/text/frontend.py) |
| Fix **money / units / dates / phone numbers / URLs** | The matching pass in [text_norm.py](src/common/text/text_norm.py) (section 4.1) |
| Fix **Urdu / English numbers** | [urdu_numbers.py](src/common/text/urdu_numbers.py) / `english_numbers` in [frontend.py](src/common/text/frontend.py#L216) |
| Change whether a word counts as **English or a Roman-Urdu name** | [names.py](src/common/text/names.py#L201) (`route`); `ROMAN_URDU_EXTRA` in frontend.py |
| See **exactly what the model reads** for a text | `python src/common/text/pk_gazetteer.py --prepare "…"` |
| Change **speaking speed / steps / temperature** | `speak.py` flags; `demo/server.py` `Voice.speak` |
| Change **sentence splitting** for long answers | `sentences` / `ABBR_END` in [demo/server.py](demo/server.py#L46) and `sentences` in [test_page.py](src/5_evaluate/test_page.py#L42) |
| Change **which clips the gate keeps** | `VOICE_MIN`, `ANAM_URDU_CER_LINE`, `is_loop` in [gate.py](src/1_gate/gate.py#L47) |
| Change **loudness matching / held-out video / row B** | [build_voice.py](src/2_build_lists/build_voice.py) (`PEAK`, `--held`, `--p-mix`) |
| Change **training hyperparameters** | [train.sh](src/3_train/train.sh), [train.py](src/3_train/train.py#L106) |
| Change **how phone embeddings are initialised** | `INIT_FROM` in [phone_init.py](src/3_train/phone_init.py), `SEED_V2` in [train.py](src/3_train/train.py#L39) |
| Change **how speakers are injected** into the model | [spk_cond.py](src/common/model/spk_cond.py) |
| Change the **pronunciation-test pass threshold / word groups** | `PASS_D`, `PRIMARY`, `PAIRS` in [pron_eval.py](src/5_evaluate/pron_eval.py#L63) |
| Change **de-muffle strength / iterations** | `--demuffle`, `ITERS`, `STEP`, `CAP` in [voice_edit.py](src/4_demuffle/voice_edit.py#L48); the reference clips in [demuffle.py](src/4_demuffle/demuffle.py) |
| Move any data / model file | [src/paths.py](src/paths.py) |

**Remember:** a text-rule change affects speaking immediately. The training lists keep the old text until step 2 is
re-run and the model is retrained. Run both front-end test suites after any edit:
```bash
python src/common/text/text_norm.py --test && python src/common/text/pk_gazetteer.py --test
```
