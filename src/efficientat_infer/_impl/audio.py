"""Own EfficientAT file/waveform normalization and optional right padding.

Mono averaging and padding precede sinc resampling, matching the evaluated
input path. Window splitting is a caller policy, never a hidden package step.
Reads: SoundFile file metadata, NumPy/Torch waveforms, sibling resampler.
"""

import math
from pathlib import Path

import numpy as np
import torch

from .resample import resample


def normalize(audio, sample_rate, device, *, pad_to_seconds=None):
    if isinstance(audio, (str, Path)):
        if sample_rate is not None:
            raise ValueError("File input uses its own sample rate; omit sample_rate")
        import soundfile as sf

        data, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T.copy())
    else:
        if type(sample_rate) is not int or sample_rate <= 0:
            raise ValueError("Waveform input requires a positive integer sample_rate")
        if isinstance(audio, np.ndarray):
            if audio.dtype.kind != "f":
                raise ValueError("Waveform must contain floating-point samples")
            waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
        elif isinstance(audio, torch.Tensor):
            waveform = audio.detach()
        else:
            raise TypeError("audio must be a path, NumPy array, or Torch tensor")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or min(waveform.shape) < 1:
        raise ValueError("Waveform shape must be [samples] or [channels, samples]")
    if not waveform.is_floating_point():
        raise ValueError("Waveform must contain floating-point samples")
    waveform = waveform.to(device=device, dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise ValueError("Waveform contains non-finite samples")
    channels, input_samples = waveform.shape
    waveform = waveform.mean(dim=0)
    padding_samples = 0
    if pad_to_seconds is not None:
        if (
            isinstance(pad_to_seconds, bool)
            or not isinstance(pad_to_seconds, (int, float))
            or not math.isfinite(pad_to_seconds)
            or pad_to_seconds <= 0
        ):
            raise ValueError("pad_to_seconds must be a positive finite number")
        target = round(pad_to_seconds * sample_rate)
        if target < input_samples:
            raise ValueError("pad_to_seconds is shorter than the supplied audio")
        padding_samples = target - input_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding_samples))
    if sample_rate != 32000:
        waveform = resample(waveform, sample_rate, 32000)
    if waveform.numel() <= 513:
        raise ValueError("Audio is too short for the EfficientAT mel frontend; pad explicitly")
    info = {
        "sample_rate": sample_rate,
        "channels": channels,
        "input_samples": input_samples,
        "input_duration_seconds": input_samples / sample_rate,
        "analysis_sample_rate": 32000,
        "analysis_channels": 1,
        "model_samples": waveform.numel(),
        "padding_samples": padding_samples,
        "padding_seconds": padding_samples / sample_rate,
        "pad_to_seconds": pad_to_seconds,
    }
    return waveform.unsqueeze(0), info
