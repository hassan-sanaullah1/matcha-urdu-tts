"""The vocoder: turns the mel spectrogram Matcha produces into 22,050 Hz audio.

It is BSC-LT/vocos-mel-22khz, loaded from models/vocoder/ (a local copy, so no download is needed).
Copied from finetune/src/inference/matcha_tts.py (_load_vocos_22k); only the file locations changed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import paths  # noqa: E402  -- kiosk-tts/src/paths.py: every location + the import path

import torch


def _load_vocos_22k():
    """Build vocos-mel-22khz from its backbone+head only.

    Vocos.from_pretrained() also instantiates the mel feature extractor, whose
    signature changed between vocos releases (no f_min/f_max here). We feed mel
    straight from Matcha, so the extractor is dead weight -- skip it.
    """
    import yaml
    from vocos.models import VocosBackbone
    from vocos.heads import ISTFTHead

    cfg = yaml.safe_load(open(os.path.join(paths.VOCODER, "config.yaml"), encoding="utf-8"))
    backbone = VocosBackbone(**cfg["backbone"]["init_args"])
    head = ISTFTHead(**cfg["head"]["init_args"])
    sd = torch.load(os.path.join(paths.VOCODER, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()
                              if k.startswith("backbone.")})
    head.load_state_dict({k[len("head."):]: v for k, v in sd.items()
                          if k.startswith("head.")})
    backbone.eval(); head.eval()

    class _V:
        def decode(self, mel):
            return head(backbone(mel))
    return _V()
