# src: the code

Read the numbered folders in order — they are the pipeline, start to finish, for fine-tuning Matcha-TTS into a
voice. `common/` holds the code every step shares. The pipeline is **voice-agnostic**: every script's defaults run
the shipped voice, and every script takes flags to run on a voice of your own instead (see
[README: fine-tune on your own dataset](../README.md#fine-tune-on-your-own-dataset)).

**How imports work.** Each script's first lines find `src/paths.py` and import it. That puts every folder below on
Python's import path, so any file can `import pk_gazetteer` or `import spk_cond` wherever it lives. `paths.py` also
holds every file location (models, data, checkpoints, results), so no script hard-codes a path.

**How to run.** From the `kiosk-tts` folder: `python src/<folder>/<script>.py --help`. Every script's docstring (the
text at its top) says what it does, what it reads and writes, and gives the exact command.

## Pipeline steps

| Folder | Script | What it does | Reads → writes |
|---|---|---|---|
| `1_gate/` | `gate.py` | **The shipped dataset's own step 1.** Keeps the clips that are clearly its speaker with clean Urdu (TitaNet voice check + an independent Omnilingual Urdu check). | `data/1_source/voice` → `data/2_clean/voice_clean` |
| | `import_voice.py` | **Step 1 for ANY other dataset.** Decodes any audio format to 22,050 Hz mono, matches audio to its transcript, groups clips into recordings, writes the same `manifest.jsonl` shape as `gate.py`. `--gate` adds the same TitaNet/Omnilingual checks. | your audio + transcript → `data/2_clean/<name>` |
| | `gate_utils.py` | The quality-check helpers both of the above can use (Omnilingual ASR, Urdu-only error score, TitaNet embedder) | |
| `2_build_lists/` | `build_voice.py` | **Voice-agnostic.** Any target speaker's clips → Matcha filelists (speaker 0): loudness-match to the base voice, run the text front end, hold out one recording, write `train/val/test.txt`. Defaults to the shipped voice; point `--clean` at any other. | `data/2_clean/<name>` → `data/3_training_lists/<name>` |
| | `build_rasa.py` | The Rasa Urdu studio clips → an optional support list (speaker 1): the same text pipeline, own loudness match. See [README: fine-tune on your own dataset](../README.md#fine-tune-on-your-own-dataset). | `data/1_source/rasa` → `data/3_training_lists/rasa_female` |
| `3_train/` | `train.py` | **Voice-agnostic.** Fine-tunes the base model on any target voice (speaker 0), optionally plus a support voice (speaker 1), with small speaker vectors. `--support` is optional — see the README. | `models/base`, the lists → `checkpoints/<tag>/`, `logs/<tag>/` |
| | `train.sh` | The exact recipe used for the shipped voice, every flag explained (8,000 steps, with Rasa support) | |
| | `phone_init.py` | Which Urdu letter each English phone symbol's embedding starts from | |
| `4_demuffle/` | `demuffle.py` | **Optional, voice-agnostic.** Lifts a trained voice's clarity toward its own real recordings, by editing the decoder's last layer in closed loop. No retraining; skip it if the voice already sounds fine. | a trained checkpoint + its lists → `checkpoints/<name>/` |
| | `voice_edit.py` | The shared weight-edit engine `demuffle.py` drives (`--demuffle`; it also knows `--swap/--blend/--pace`, unused by demuffle.py) | |
| `5_evaluate/` | `pron_eval.py` | **Main accuracy score.** A phone recogniser hears each target word (knowledge-base words, acronyms, minimal pairs such as cooled/cold). | checkpoint → `results/pron_<name>/` |
| | `phone_check.py` | The phone recogniser and aligner behind `pron_eval`; run alone, it audits the training labels against the audio | |
| | `make_open_suite.py` | The word lists and knowledge-base acronyms used when `pron_eval --rebuild` regenerates the suite | knowledge base → `data/4_eval/` |
| | `eval_target.py` | Urdu character error rate (Whisper), an English word check, voice distance, and a listening page | checkpoint → `results/eval_<name>/` |
| | `test_page.py` | Kiosk listening page: 30 knowledge-base answers and 143 names, with right/wrong buttons | checkpoint → `results/<name>/` |
| `6_speak/` | `speak.py` | Turns one text into one wav, exactly as the kiosk speaks it | text → wav |

The type-and-listen web page is `demo/server.py` (+ `demo/index.html`), which loads a checkpoint and speaks typed text.

## common/ (shared code)

**`common/text/`: the text front end.** It turns any chatbot text into the symbols the model reads. Training and
speaking both call the same function, `pk_gazetteer.prepare_pk(text)`, so they can never disagree.

| File | What it does |
|---|---|
| `pk_gazetteer.py` | **Entry point, `prepare_pk`.** Pakistani names, places and companies written in Latin → Urdu script. English sound fixes (British vowels). `--test` runs its tests. |
| `text_norm.py` | Cleans what an LLM writes: markdown, links, money, units, dates, decimals, phone numbers, ranks, codes and acronyms. `PK_PHONES` gives exact sounds for names. `--test` runs its tests. |
| `frontend.py` | Splits Urdu from English, spells acronyms, reads English numbers (NeMo), and guards Roman-Urdu words |
| `phones.py` | English word → Pakistani-English phone symbols (espeak-ng + mapping tables) |
| `names.py` | Decides whether a Latin word is English or a Roman-Urdu name (word lists, WordNet, transliteration pairs) |
| `en_g2p.py` | Small English spelling fixes before espeak |
| `text_ipa.py` | The Roman-Urdu table used by `frontend.py` |
| `urdu_text_norm.py`, `urdu_numbers.py` | Urdu normalisation and Urdu number words |
| `tts.py` | The vocabulary (`vocab`), `clean`, `encode`, loudness `normalize`, and `synth` (one text → audio) |

**`common/model/`: the model.**

| File | What it does |
|---|---|
| `matcha_torch.py` | Builds Matcha-TTS from a checkpoint, and loads the vocabulary and vocoder (all from `models/`) |
| `spk_cond.py` | Adds up to two speaker vectors to the single-speaker base model without reshaping it. `load_for_inference(ckpt, device, spk=0)` loads the target voice (speaker 0). |
| `vocoder.py` | The Vocos vocoder (mel spectrogram → 22,050 Hz audio) from `models/vocoder/` |

**`common/dataset/dataset_utils.py`:** small helpers used by the list-building step (loudness, soft limiter, row B).

## Where the code came from

It was copied out of `~/Desktop/kaani`, where 26 rounds of experiments live, and depends on none of it:
- `pk-bilingual/src`: `train.py`, `text_norm`, `pk_gazetteer`, `spk_cond`, the list builders;
- `urdu-phones/src`: `frontend`, `phones`, `names`, `tts`, `phone_init`;
- `finetune/src/inference`: `matcha_torch`, `vocoder`, `urdu_text_norm`, `urdu_numbers`;
- `pk-voice/src`: `text_ipa`, `en_g2p`;
- an earlier kaani fine-tune round: `voice_edit`;
- a later kaani fine-tune round: `gate` (was `prep.py`), `demuffle`, and the recipe in `train.sh`.

`import_voice.py` is new: it was written for this repo, to bring in a dataset that isn't shaped like the shipped
one.

Only file locations were changed and unused options removed. The original project verified that the text front end
gives identical output on 2,956 texts, that `speak.py` is bit-identical to the original, that training rebuilds the
exact training rows, and that the shipped voice (lists, training rows, de-muffle edit) rebuilds identically.
