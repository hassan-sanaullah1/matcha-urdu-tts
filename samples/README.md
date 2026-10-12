# samples: what the shipped voice sounds like

Open **`index.html`** in a browser. It plays the same sentences spoken by the shipped voice
(`checkpoints/voice_demuffled/model.ckpt` — 8k steps, de-muffled, its own timing), and in the last section
puts each held-out sentence next to the speaker's **real recording** so you can judge how close the voice is.

| Folder | What it holds |
|---|---|
| `voice/` | The shipped voice speaking the sample sentences |
| `voice_real_recordings/` | The speaker's real recordings of the 8 held-out sentences, for comparison (from `data/1_source/voice`) |

Inside `voice/`:
- `kiosk/`: 24 MARI knowledge-base answers (`mari00`–`mari23`), 4 hard Urdu lines (`urdu0`–`urdu3`) and 6
  acronyms (`acr_*`).
- `hard_page/`: the 90-sentence hard page (`A00`–`G29`), groups A–G (titles are the section headings in `index.html`).
- `held_out/`: 8 sentences from the held-out video, played next to the real recordings.

`sentences.tsv` lists every item with its text; `index.html` is generated from it.

**Where they came from.** All files are mp3, rendered at the kiosk settings (32 steps, temperature 0.35) from an
earlier kaani fine-tune round's listening pages. To make new ones from a checkpoint:
`python src/6_speak/speak.py "<text>" -o out.wav`.
