"""Compare a real audio file end to end with pinned upstream EfficientAT.

This development-only A/B requires TorchAudio and TorchVision in its environment.
Reads: original source checkout, author weights and a caller-provided audio file.
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    from efficientat_infer import EfficientATSession
    from efficientat_infer._impl.audio import normalize
    from efficientat_infer._impl.frontend import EfficientATMel

    torch.set_num_threads(2)
    os.chdir(args.upstream)
    sys.path.insert(0, str(args.upstream))
    from models.mn.model import get_model
    from models.preprocess import AugmentMelSTFT

    data, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) > rate * 10:
        raise ValueError("This A/B expects audio at most ten seconds long")
    x = np.pad(mono, (0, rate * 10 - len(mono)))
    upstream_wave = AF.resample(torch.from_numpy(x), rate, 32000)[None, :]
    package_wave, _ = normalize(args.audio, None, torch.device("cpu"), pad_to_seconds=10)
    wave_diff = (upstream_wave - package_wave).abs().max().item()
    with torch.inference_mode():
        upstream_mel = AugmentMelSTFT(freqm=0, timem=0).eval()(upstream_wave)
        package_mel = EfficientATMel()(package_wave)
    mel_diff = (upstream_mel - package_mel).abs().max().item()
    print("input", args.audio, "wave_diff", wave_diff, "mel_diff", mel_diff)
    for name, width, filename in [
        ("mn04", 0.4, "mn04_as_mAP_432.pt"),
        ("mn10", 1.0, "mn10_as_mAP_471.pt"),
    ]:
        with contextlib.redirect_stdout(io.StringIO()):
            upstream = get_model(width_mult=width, pretrained_name=None)
        state = torch.load(args.weights / filename, map_location="cpu", weights_only=True)
        upstream.load_state_dict(state, strict=True)
        upstream.eval()
        with torch.inference_mode():
            logits, features = upstream(upstream_mel.unsqueeze(0))
            expected = torch.sigmoid(logits.float())[0].numpy()
        with EfficientATSession(name, device="cpu", checkpoint_dir=args.weights, offline=True) as session:
            session.load()
            result = session.infer(args.audio, pad_to_seconds=10)
        score_diff = np.max(np.abs(result.scores - expected))
        feature_diff = np.max(np.abs(result.embedding - features[0].numpy()))
        print(name, "score_max_abs", score_diff, "feature_max_abs", feature_diff,
              "same_top1", int(np.argmax(result.scores)) == int(np.argmax(expected)))
        assert np.allclose(result.scores, expected, atol=1e-6, rtol=1e-6)
        assert np.allclose(result.embedding, features[0].numpy(), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    main()
