"""The single canonical Urdu text normalisation, shared by training and inference.

This exists because there were two of them and they disagreed. matcha_torch.encode
mapped characters straight through (`stoi.get(ch, space)`), while the training
cleaner additionally NFC-normalised, collapsed runs of whitespace and stripped. On
the 20 clean eval sentences the two agreed exactly, so the difference was invisible
-- but real corpus text contains double spaces, and there the two paths produced
sequences of different LENGTH. Training would then optimise against a tokenisation
the model never sees at inference.

Any future change to normalisation belongs here and nowhere else, so the two paths
cannot drift apart again. verify_pretrain.py check [1] fails loudly if they do.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import unicodedata


def urdu_norm(text, vocab):
    """NFC-normalise, replace out-of-vocab characters with space, collapse runs.

    `vocab` is any container of the 350 in-vocabulary symbols. Out-of-vocab maps to
    space rather than raising, matching what the shipped ONNX inference path does.
    """
    text = unicodedata.normalize("NFC", str(text))
    out, prev_space = [], False
    for ch in text:
        c = ch if ch in vocab else " "
        if c == " ":
            if not prev_space:
                out.append(c)
            prev_space = True
        else:
            out.append(c)
            prev_space = False
    return "".join(out).strip()


# ---------------------------------------------------------------------------
# Symbol hygiene
#
# urdu_norm() above partitions characters two ways: in vocab, or not. That is
# not enough, because being in the vocab does not mean the embedding row was ever
# trained. symbol_audit.py measures this directly against the checkpoint -- the
# encoder was trained with weight_decay=0.0, so a row that never received gradient
# sits exactly at its normal_(0, 192**-0.5) initialisation, and the Latin+IPA
# block (ids 24-183) matches that init to five significant figures.
#
# The blocks below are the audit's verdicts (KS vs that empirical null):
#
#   ids  24-183  Latin A-Z a-z + IPA   untrained by construction (reference block)
#   ids 275-284  digits U+06F0-06F9    TRAINED   (p = 5e-15)
#   ids 285-294  digits U+0660-0669    UNTRAINED (p = 0.94)   <- visually identical
#   ids 324-349  Bengali               trained, but never legitimately emitted
#
# The two digit blocks are the dangerous one: they render near-identically, both
# are in the vocab, and only one of them means anything to the model.
#
# The rewrites below are chosen so that they are correct whether or not the
# trainedness call for a given row is right. Mapping ASCII '?' to Urdu '؟' is the
# correct normalisation for Urdu text regardless; deleting a zero-width joiner is
# correct regardless. Nothing here depends on winning a low-power statistical
# argument about a single embedding row.
# ---------------------------------------------------------------------------

# Untrained id ranges, as [lo, hi) pairs. Latin is listed but handled separately,
# because the whole point of the code-switching work is to make it trained.
_UNTRAINED_RANGES = [(24, 184), (285, 295), (324, 350)]

# char -> replacement. "" deletes; a string substitutes.
_REWRITE = {
    # ASCII punctuation -> the Urdu forms, which are the frequent, certainly
    # trained ones. ASCII '?' in particular scores z=-0.84 against the untrained
    # reference and cannot plausibly have occurred in an Urdu-only studio corpus.
    "?": "؟", ".": "۔", ",": "،", ";": "؛",
    # Zero-width control characters. These MUST delete rather than become a space:
    # they are zero-width, so turning them into a space would insert a spurious
    # word boundary and change the prosody. They survive NFC and are in the vocab,
    # so urdu_norm alone passes them straight through into the model.
    "‌": "", "‍": "", "‎": "", "‏": "",
    # Tatweel is a pure glyph-elongation character with no phonetic content.
    "ـ": "",
    # Brackets and quote marks have no reliable acoustic realisation; a space is
    # the honest rendering of the pause they imply.
    "(": " ", ")": " ", '"': " ", "«": " ", "»": " ", "“": " ", "”": " ",
    "'": " ", "‘": " ", "’": " ", "[": " ", "]": " ",
    # NOTE: the apostrophe entries above are overridden to "" (delete, no space)
    # when latin="keep" -- see _APOSTROPHE in normalize_for_tts. In Urdu-only text
    # an apostrophe is a quote mark and a space is right; in English it is a
    # contraction marker, and a space turns "don't" into "don t", inserting a word
    # boundary and a pause in the middle of a single spoken word.
    "¡": " ", "¿": " ", "—": " ", "–": " ", "-": " ", "…": " ۔ ",
    # Honorific ligatures, spelled out as they are actually read aloud.
    "ﷺ": " صلی اللہ علیہ وسلم ",   # ﷺ
    "ﷻ": " جل جلالہ ",             # ﷻ
    "﷽": " بسم اللہ الرحمن الرحیم ",  # ﷽
}

# Arabic-Indic digits (untrained block) -> extended Arabic-Indic (trained block).
# expand_numbers() normally consumes these first; this is the backstop for any
# that survive, so an untrained row is never emitted.
for _i in range(10):
    _REWRITE[chr(0x0660 + _i)] = chr(0x06F0 + _i)


def trained_symbols(vocab):
    """The subset of `vocab` whose embedding rows pretraining actually trained.

    `vocab` must be the id-ordered symbol mapping (char -> id), so the ranges from
    symbol_audit.py line up. Latin is excluded here; callers that have a checkpoint
    which HAS been trained on Latin pass latin="keep" to normalize_for_tts.
    """
    bad = set()
    for lo, hi in _UNTRAINED_RANGES:
        bad.update(range(lo, hi))
    return {ch for ch, i in vocab.items() if i not in bad}


def normalize_for_tts(text, vocab, latin="drop", expand_nums=True):
    """Full inference-time text preparation: hygiene, then urdu_norm.

    Guarantees, when latin="drop", that every character surviving into the
    tokenizer has a trained embedding row. That is a strictly stronger property
    than urdu_norm's "is in the vocab", and it is asserted by
    assert_trained_only().

    latin:
      "drop" -- Latin letters become spaces. Correct for the base checkpoint,
                where ids 24-75 are at random initialisation: deleting an English
                word is bad, but strictly better than feeding the encoder noise,
                which is what happens today.
      "keep" -- Latin letters pass through. For a checkpoint that has been trained
                to read them; wrong for the base model.

    urdu_norm() is still the last step and is unchanged, so train/inference token
    parity (verify_pretrain.py check [1]) is preserved by construction.
    """
    import unicodedata

    text = unicodedata.normalize("NFC", str(text))

    if expand_nums:
        from urdu_numbers import expand_numbers
        text = expand_numbers(text)

    # An apostrophe between two letters is a contraction, not a quote. Mapping it
    # to a space (the Urdu-correct default) yields "don t" / "cow s": the model is
    # taught a word boundary, and the pause that goes with it, inside one spoken
    # word. Deleting it instead gives "dont" / "cows", whose length and prosody
    # match the audio. Only applied under latin="keep", so Urdu-only text is
    # untouched.
    rewrite = _REWRITE
    if latin == "keep":
        rewrite = {**_REWRITE, "'": "", "’": "", "‘": ""}

    out = []
    for ch in text:
        if ch in rewrite:
            out.append(rewrite[ch])
        elif latin == "drop" and ch.isascii() and ch.isalpha():
            out.append(" ")
        else:
            out.append(ch)
    text = "".join(out)

    return urdu_norm(text, vocab)


def assert_trained_only(text, vocab, latin="drop"):
    """Raise if `text` would tokenise to any untrained embedding row.

    Cheap enough to run on every synthesis. The failure it catches is silent --
    an untrained row produces plausible-sounding noise, not an error -- so it has
    to be asserted rather than hoped for.
    """
    ok = trained_symbols(vocab)
    if latin == "keep":
        ok |= {ch for ch, i in vocab.items() if 24 <= i < 76}
    bad = sorted({ch for ch in text if ch in vocab and ch not in ok})
    if bad:
        raise ValueError(
            "text would emit untrained embedding rows: "
            + ", ".join(f"{ch!r}(id={vocab[ch]})" for ch in bad)
        )
    return True
