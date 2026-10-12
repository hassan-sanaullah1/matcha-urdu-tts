# data

The folder numbers match the pipeline steps in `src/`. Every path stored inside these files is relative to the
`kiosk-tts` folder, so run the scripts from there.

```
1_source ─(src/1_gate or src/1_gate/import_voice.py)─► 2_clean ─(src/2_build_lists)─► 3_training_lists ─(src/3_train)─► checkpoints/
                                                                                              4_eval ─(src/5_evaluate)
```

The shipped voice's audio is used exactly as delivered by `~/Desktop/urdu-dataset-builder` (de-musicked, cut in
silences, gate cuts repaired); nothing in this folder re-cuts or alters it. Bringing your own voice? See
[README: fine-tune on your own dataset](../README.md#fine-tune-on-your-own-dataset) — the same folder numbers apply,
with your own data under a name of your choosing instead of `voice`.

## 1_source: the raw material (read-only inputs)

| Folder | What it holds | Size |
|---|---|---|
| `voice/` | **The shipped voice's dataset**, from `~/Desktop/urdu-dataset-builder`: `wav/` (864 clips, 1.74 h, 22,050 Hz mono), `train.txt`, `val.txt`, `metadata.csv` (every check score per clip), `README.md` (the dataset's own notes), `report.txt`. Its unused review pool was left out. | 266 MB |
| `rasa/*.parquet` | [ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) Urdu shards 1–4 (studio speech with text). The female voice is the **support speaker** (id 1): it steadies rare Urdu sounds. Optional for a new voice. | 1.1 GB |
| `anchor/*.wav` | 50 clips of the base model's own training voice, used as the loudness target: every training clip is gain-matched to them. Needed by both build scripts in step 2. | 18 MB |

## 2_clean: which clips to keep (step 1's result)

| Folder | What it holds |
|---|---|
| `voice_clean/` | The output of `src/1_gate/gate.py`: `manifest.jsonl` (all 864 clips with their voice and text scores), `drop.txt` (82 clips left out: 36 auto-released review clips, 35 Urdu mismatches, 15 not clearly the target voice; some fail more than one), `keep_summary.json`, `urdu_check.jsonl`, `titanet_centroid.npy`. `wav` is a link to `1_source/voice/wav`. A voice you import with `import_voice.py` writes the same shape here, under the name you give it. |

## 3_training_lists: what training reads

Each `*.txt` line is `<wav path>|<text the model reads>`. The text is Urdu in Urdu script plus English as
Pakistani-English phone symbols, made by `pk_gazetteer.prepare_pk`. Every clip has two lines: row A is the normal
text; row B has some Urdu words written as phones. The model trains on row A only (`--no-row-b`).

| Folder | What it holds |
|---|---|
| `voice/` | **The exact lists the shipped model was trained on** (speaker 0): 760 training clips (1.53 h), plus 10 val and 12 test clips from the held-out video `-gI_ALIlkR8`. `wavs/` holds the clips after loudness matching. |
| `rasa_female/` | **The exact list used for speaker 1**: Rasa female studio Urdu, 893 clips (1.51 h), shuffled. Training picks ~325 rows from it (30% of each epoch). Optional for a new voice. |

Rebuilding with step 2 writes new folders (`voice_rebuild/`, `rasa_female_rebuild/`) and never overwrites these.
The Rasa rebuild matches exactly; the shipped voice's rebuild reproduces the same clips, with text from today's
front end (which is also what runs at speaking time).

## 4_eval: frozen test sets (step 5)

| File | Used by |
|---|---|
| `knowledge_base/mari_energies_knowledge_base.md` | **The MARI Energies knowledge base**: the chatbot's source text and the main pronunciation target. The text rules (`src/common/text/text_norm.py`, `pk_gazetteer.py`) were written from it. `pron_eval --rebuild` and `make_open_suite.py` take their words, acronyms and sentences from it. |
| `knowledge_base/showcase_*.txt` | The knowledge-base showcase answers (code-switched), also part of the pronunciation suite |
| `pron_suite.json` | `5_evaluate/pron_eval.py`: 3,277 renders — knowledge-base words, acronyms and sentences, minimal pairs, and general words. The shared frozen suite every checkpoint is scored on. |
| `open_suite.json` | `5_evaluate/make_open_suite.py`: 372 open-vocabulary lines (general English and Urdu, MARI lines, all 180 knowledge-base acronyms) |
| `eval_suite.json` | `5_evaluate/eval_target.py`: Urdu (parity and rare sounds), English, code-switched, and numbers |
| `kb_test_sentences.txt`, `kb_test_names.txt` | `5_evaluate/test_page.py`: 30 kiosk answers and all 143 knowledge-base names |
| `shahed_ref/` | `eval_target.py`'s voice-distance reference (48 clips of an earlier reference speaker). A legacy anchor: `sim_shahed` is a between-checkpoint number, not "is this the shipped voice" — for that, use the TitaNet check in `docs/training_report.md`. |
