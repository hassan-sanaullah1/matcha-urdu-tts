# Round 16: the shipped voice, trained like r5_p2 (1 Oct 2026, overnight)

**What you asked for:**
- Fine-tune on `urdu-transcriber's dataset for this voice`.
- The voice should be hers, with r5_p2's accuracy and no choppiness.
- Also: "check which duration would be better to use in the duration predictor".

**Short answer:**
- **The voice is hers, and the accuracy is at or above r5_p2 everywhere.**
  - KB words 76% vs 74%.
  - General English words 74–76% vs 67%.
  - Hard-page English 75–76% vs 65%.
  - Pure Urdu level (Urdu error 0.05 vs 0.06).
- **The choppiness measures match her real recordings,** the best of any model so far. Your ear decides, as always.
- **Durations: keep the timing the model learns itself.**
  - r5_p2's timing lowers accuracy (acronyms 77.8% → 73.3%).
  - Her timing alone is the same as the model's own, because her clips already make up 70% of what it learns from.
- **Weak spot: it is duller than her real recordings.** The 2–4 kHz clarity is 0.9 vs her 2.1. The de-muffle edit lifts the 6k model to the 8k level but no further.

**Listen:**
- Page: (an earlier kaani fine-tune round's listening page)
- Type-and-listen demo (8k, de-muffled): http://127.0.0.1:8024

## What was done

1. **Data (src/prep.py).** The dataset is used exactly as delivered: no audio altered here. The clean-clip folder is a symlink to its `wav/`.
   - **Already processed upstream:** the dataset was de-musicked with MossFormer2, and gate dropouts were repaired in 632 clips (see its README).
   - **Round 13's gate:**
     - an independent Omnilingual Urdu check against the Anam line;
     - TitaNet similarity to her centroid of at least 0.70;
     - the loop rule;
     - the 36 auto-released review clips dropped (the README's strict option);
     - re-decode grade C (none).
   - **Kept: 782 of 864 clips.** 1.53 h for training, about the same as Anam's 1.51 h.
   - **Her transcripts are cleaner than Anam's:** Urdu CER median 0.038 vs 0.075.
   - **Held out:** the dataset's own validation video (22 clips).
2. **Build** (pk-bilingual `build_filelists.py`, unchanged).
   - One uniform gain of 0.31 (her audio is very loud, −10 dBFS) to the base model's level, as for every earlier voice.
   - 760 training clips; 98% mixed Urdu/English.
3. **Training (src/train.py):** r5_p2's exact flags.
   - The parity check passed; only data, tag and steps differ.
   - Her voice is speaker 0, with Rasa as 30% support. 8,000 steps took 41 minutes.
4. **Tests.**
   - pron_eval: 3,277 words, paired against r5_p2.
   - The 90-sentence hard page.
   - The Round 10 choppiness diagnosis on her held-out clips.
   - Voice (TitaNet) and clarity.
   - Run at 6k and 8k. The step with the best KB words + acronyms (6k) went on to the timing test.
5. **Durations (src/timing.py).**
   - The duration predictor has no speaker input. The mix of clips it trains on decides whose timing it learns.
   - Three timings were compared on the same model: its own; r5_p2's (Anam 70% / Rasa 30%, injected); and hers only (the predictor retrained on her clips).
6. **De-muffle (src/demuffle.py):** Round 13b's closed-loop weight edit, fitted on 60 of her training clips and tested on her held-out video.
   - 6k: band-shape gap 1.23 → 0.81 dB.
   - 8k: 0.91 → 0.77 dB.

## Results

| Model | KB words | KB acronyms | general words | general acronyms | hard page English | hard page Urdu error | tone kept | voice is hers (≥ 0.8) |
|---|---|---|---|---|---|---|---|---|
| r5_p2 (Anam, today) | 74.3% | 75.6% | 67.4% | 82% | 65.2% | 0.130 | 87% | – |
| B4 (Rasa voice) | 74.1% | 73.3% | 67.8% | 82% | 66.8% | 0.167 | 98% | – |
| **new, 6k, own timing** | 76.2% | **77.8%** | 74.3% | 84% | **76.3%** | 0.153 | 101% | 1.00 |
| new, 6k, r5_p2's timing | 75.6% | 73.3% | 71.6% | 80% | 75.1% | 0.148 | 101% | 1.02 |
| new, 6k, her timing only | **76.7%** | 75.0% | 72.8% | 84% | 76.1% | **0.144** | 101% | 1.01 |
| new, 6k, de-muffled | 75.9% | 75.6% | 73.9% | 84% | 75.7% | 0.149 | 100% | 0.98 |
| new, 8k (no edit) | 76.5% | 74.4% | **75.9%** | 82% | – | – | – | 1.01 |
| new, 8k, de-muffled | 76.2% | 74.4% | 75.5% | 84% | 75.2% | 0.151 | 100% | 1.01 |

- Changes against r5_p2 are paired (same words, same seeds). KB words and general words are significantly better for every new-voice row, and KB sentences for every row except r5_p2's timing.
- **Acronyms:** in the 6k own-timing row they are significantly better (+2.2, interval excludes 0). Elsewhere they are within noise.
- **Pure Urdu (hard page group D):** 0.051–0.054 vs r5_p2's 0.062.
- **Choppiness diagnosis** (her held-out clips, against her real recordings):
  - tone strength 100–101% of hers (r5_p2: 87%, Mahwash: 95%);
  - voicing breaks 0.020/s vs her 0.023;
  - no cut-off sentence ends (r5_p2: 26%).
  - Only 10 held-out clips, so treat this as indicative.
  - The diagnosis renders at her real timing, so it cannot tell the three timings apart.

**Durations, on her 10 held-out clips against her real alignment:**

| Timing | error per sound | total length |
|---|---|---|
| own | 1.18 frames | 12% slower than her |
| r5_p2's | 1.30 frames | 2% slower |
| hers only | 1.18 frames | 13% slower |

- In real speech 54% of her word gaps have no pause; every timing gives 0–1%. That is the Round 14 finding again: a timing model trained on averages gives every word a small gap.

**Clarity, the same 22 held-out sentences:**

| | 2–4 kHz clarity |
|---|---|
| her real recordings | 2.07 |
| 6k | 0.61 |
| 6k de-muffled | 0.90 |
| 8k | 0.93 |
| 8k de-muffled | 0.90 |

- r5_p2 keeps 65% of Anam's clarity; this voice keeps about 44% of hers.
- Her recordings are unusually bright (MossFormer2 plus heavy compression). Her training videos range from 1.0 to 2.9 (mean 1.8).
- The de-muffle matched her average band shape on the fit clips (0.05 dB). The remaining gap is in the loud 2–4 kHz peaks, which the model smooths. A bias cannot add those.

## Recommendation

- **Durations:** keep the model's own timing.
  - r5_p2's timing costs acronyms and general words.
  - Hers-only gives the same as own.
- **Candidates for your ear:**
  - 6k own timing: the best acronyms.
  - 8k de-muffled: the clearest at the same accuracy; this is in the demo on 8024.
  - Both are on the page.
- r5_p2 stays in the kiosk until you pick.

## Files

- **src/:** prep.py, train.py, evaluate.py, timing.py, demuffle.py, page.py.
- **logs/:** run_chain.sh, run_demuffle.sh, run_dm8.sh.
- **checkpoints/:**
  - r16_voice: 2k–8k;
  - r16_her_timing;
  - edits/dm (6k de-muffled);
  - edits/dm8 (8k de-muffled).
- **output/:** page, pron_*, hard_*, diag_*, voice_*.json, timing_r16_voice.json.
- **data/:** voice_clean (manifest + symlink), voice_build, urdu_check.jsonl, drop.txt, keep_summary.json.
