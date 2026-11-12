"""Capture a deterministic, nontrivial baseline from original EfficientAT.

Run with an environment containing upstream TorchAudio and TorchVision.
Reads: the pinned upstream checkout and author release checkpoints supplied by CLI.
"""

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def fixture_waveform() -> np.ndarray:
    rate = 32000
    t = np.arange(rate * 10, dtype=np.float64) / rate
    chirp = np.sin(2 * np.pi * (180 * t + 95 * t * t))
    bass = np.sin(2 * np.pi * 73 * t)
    noise = np.random.default_rng(20261112).standard_normal(t.size)
    envelope = np.exp(-((t - 0.8) / 0.18) ** 2) + 0.8 * np.exp(-((t - 3.1) / 0.26) ** 2)
    waveform = (0.17 * chirp + 0.12 * bass + 0.10 * noise * envelope).astype("float32")
    waveform[rate * 7 :] = 0.0  # real silence/padded-tail failure regime
    return waveform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(args.upstream)
    sys.path.insert(0, str(args.upstream))
    from models.mn.model import get_model
    from models.preprocess import AugmentMelSTFT

    torch.set_num_threads(2)
    waveform = fixture_waveform()
    frontend = AugmentMelSTFT(freqm=0, timem=0).eval()
    with torch.inference_mode():
        mel = frontend(torch.from_numpy(waveform)[None, :]).numpy()
    data = {"waveform": waveform, "mel": mel}
    for name, width, filename in [
        ("mn04", 0.4, "mn04_as_mAP_432.pt"),
        ("mn10", 1.0, "mn10_as_mAP_471.pt"),
    ]:
        with contextlib.redirect_stdout(io.StringIO()):
            model = get_model(width_mult=width, pretrained_name=None)
        state = torch.load(args.weights / filename, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.inference_mode():
            logits, features = model(torch.from_numpy(mel).unsqueeze(0))
        data[f"{name}_logits"] = logits.numpy()
        data[f"{name}_features"] = features.numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    meta = {
        "upstream_revision": "a425fdce92572e602a1d5634799bd9f1f2efa806",
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
        "input": "deterministic generated waveform; 10s at 32kHz; last 3s silence",
        "checkpoint_sha256": {
            "mn04": "899a8c6217063f6941793102a2780b3dd788ab11fce468d9cf5f0577c453321c",
            "mn10": "0bd7dc2443af498c289a2e739f02ebb515d6aa3fd3ab9db539c86123ae368a4e",
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(args.output, {key: value.shape for key, value in data.items()})


if __name__ == "__main__":
    main()
