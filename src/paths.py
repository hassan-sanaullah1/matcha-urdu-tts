"""Every file location kiosk-tts uses, in one place, plus the Python import path.

Every script starts with two lines that find this file and import it:

    sys.path.insert(0, <the src/ folder>)
    import paths

Importing it puts every code folder under src/ (and the vendored Matcha-TTS) on sys.path, so any module can
import any other by its plain name (`import pk_gazetteer`, `import spk_cond`, ...) no matter which folder
it lives in. All paths below are absolute and built from this file's location, so scripts work from any
current directory (the training lists store paths relative to the kiosk-tts root, so run from there).
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))           # kiosk-tts/src
ROOT = os.path.dirname(SRC)                                 # kiosk-tts

# ---- import path: shared code first, then the pipeline steps -------------------------------------------
CODE_FOLDERS = [
    os.path.join(SRC, "common", "text"),        # text front end: any text -> the symbols the model reads
    os.path.join(SRC, "common", "model"),       # model loading, speaker tags, vocoder
    os.path.join(SRC, "common", "dataset"),     # small helpers shared by the data steps
    os.path.join(SRC, "1_gate"),                # keep the clips that are clearly the target voice with clean Urdu
    os.path.join(SRC, "2_build_lists"),         # cleaned clips -> Matcha training lists (target voice + support)
    os.path.join(SRC, "3_train"),               # fine-tune the base model on the target voice (spk 0) + support (spk 1)
    os.path.join(SRC, "4_demuffle"),            # de-muffle the trained voice through its weights only
    os.path.join(SRC, "5_evaluate"),            # pronunciation score, Urdu error, listening pages
    os.path.join(SRC, "6_speak"),               # text -> wav, as the kiosk speaks it
]
for _p in reversed(CODE_FOLDERS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Matcha-TTS source (not pip-installed; common/model/matcha_torch.py adds it to sys.path when needed)
MATCHA_SRC = os.path.join(ROOT, "vendor", "Matcha-TTS")

# ---- models --------------------------------------------------------------------------------------------
BASE_CKPT = os.path.join(ROOT, "models", "base", "matcha_urdu.ckpt")   # AhsanTalal/urdu-matcha-tts base model
BASE_VOCAB = os.path.join(ROOT, "models", "base", "vocab.json")        # its 350-symbol vocabulary
VOCODER = os.path.join(ROOT, "models", "vocoder")                      # BSC-LT/vocos-mel-22khz (mel -> audio)
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
VOICE_8K = os.path.join(CHECKPOINTS, "voice", "step=007999.ckpt")       # the shipped fine-tune, 8,000 steps
VOICE = os.path.join(CHECKPOINTS, "voice_demuffled", "model.ckpt")      # the voice in use: 8k + the de-muffle edit

# ---- lookup files read by the text front end ------------------------------------------------------------
ASSETS = os.path.join(ROOT, "assets")
NEMO_CACHE = os.path.join(ASSETS, "nemo_cache")                        # compiled NeMo English number grammars
TRANSLIT = os.path.join(ASSETS, "translit_pairs.json")                 # English <-> Urdu spelling pairs (names)
DICT_WORDS = os.path.join(ASSETS, "dict", "words")                     # English word list (was /usr/share/dict)
DICT_US = os.path.join(ASSETS, "dict", "american-english")
DICT_UK = os.path.join(ASSETS, "dict", "british-english")
EXTRA_ENGLISH = os.path.join(ASSETS, "extra_english.txt")              # extra English words (from urdu-transcriber)
NLTK_DATA = os.path.join(ASSETS, "nltk_data")                          # WordNet

# ---- data, numbered like the pipeline steps -------------------------------------------------------------
DATA = os.path.join(ROOT, "data")
SOURCE = os.path.join(DATA, "1_source")
VOICE_SOURCE = os.path.join(SOURCE, "voice")            # the shipped voice's dataset, from ~/Desktop/urdu-dataset-builder
RASA_SOURCE = os.path.join(SOURCE, "rasa")              # ai4bharat/Rasa Urdu parquet shards 1-4 (support voice)
ANCHOR = os.path.join(SOURCE, "anchor")                 # 50 base-voice clips: the loudness target
VOICE_CLEAN = os.path.join(DATA, "2_clean", "voice_clean")       # the gate's result (step 1)
TRAINING_LISTS = os.path.join(DATA, "3_training_lists")
VOICE_LISTS = os.path.join(TRAINING_LISTS, "voice")              # the target voice's lists, speaker 0 (step 2)
RASA_LISTS = os.path.join(TRAINING_LISTS, "rasa_female")         # the Rasa support list, speaker 1 (step 2)
EVAL = os.path.join(DATA, "4_eval")

# ---- outputs --------------------------------------------------------------------------------------------
RESULTS = os.path.join(ROOT, "results")                 # evaluation scores, listening pages, audits
LOGS = os.path.join(ROOT, "logs")                       # training loss logs
OUTPUT = os.path.join(RESULTS, "renders")               # single sentences spoken by speak.py
