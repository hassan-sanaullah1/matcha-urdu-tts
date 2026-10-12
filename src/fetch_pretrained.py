"""Fetch the two pretrained files this repo needs but does not commit (see .gitignore): the base
Matcha-TTS checkpoint and the Vocos vocoder weights. Both are public on Hugging Face; this just
copies them from the HF cache into the local paths every script expects (src/paths.py).

Needed for: training (step 3), speaking (step 6), the demo, and evaluation. Not needed just to
read the code or the training lists.

Usage (from the kiosk-tts folder):  python src/fetch_pretrained.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path


def main():
    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(paths.BASE_CKPT), exist_ok=True)
    if not os.path.exists(paths.BASE_CKPT):
        p = hf_hub_download("AhsanTalal/urdu-matcha-tts", "matcha_urdu.ckpt")
        shutil.copy(p, paths.BASE_CKPT)
        print(f"-> {paths.BASE_CKPT}")
    else:
        print(f"already have {paths.BASE_CKPT}")

    vocoder_bin = os.path.join(paths.VOCODER, "pytorch_model.bin")
    os.makedirs(paths.VOCODER, exist_ok=True)
    if not os.path.exists(vocoder_bin):
        p = hf_hub_download("BSC-LT/vocos-mel-22khz", "pytorch_model.bin")
        shutil.copy(p, vocoder_bin)
        print(f"-> {vocoder_bin}")
    else:
        print(f"already have {vocoder_bin}")


if __name__ == "__main__":
    main()
