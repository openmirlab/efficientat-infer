"""Eval-only EfficientAT mel frontend with checkpoint-faithful arithmetic.

Training masks and random frequency changes are deliberately absent. The
TorchAudio Kaldi mel-bank closure is retained locally under BSD-2-Clause.
Reads: mono 32 kHz Torch waveform and sibling Kaldi mel-bank function.
"""

import torch
from torch import nn

from .kaldi import get_mel_banks


class EfficientATMel(nn.Module):
    """Upstream AugmentMelSTFT in its eval configuration, without augmentation."""

    sample_rate = 32000
    n_fft = 1024
    win_length = 800
    hop_length = 320
    n_mels = 128
    fmin = 0.0
    fmax = 15000.0

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(800, periodic=False), persistent=False)
        self.register_buffer(
            "preemphasis_coefficient", torch.as_tensor([[[-0.97, 1.0]]]), persistent=False
        )
        mel_basis, _ = get_mel_banks(128, 1024, 32000, 0.0, 15000.0, 100.0, -500.0, 1.0)
        self.register_buffer("mel_basis", nn.functional.pad(mel_basis, (0, 1)), persistent=False)

    def forward(self, waveform):
        waveform = nn.functional.conv1d(
            waveform.unsqueeze(1), self.preemphasis_coefficient
        ).squeeze(1)
        spectrum = torch.view_as_real(torch.stft(
            waveform, self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, center=True, normalized=False,
            window=self.window, return_complex=True,
        ))
        power = (spectrum ** 2).sum(dim=-1)
        # Upstream disables autocast for the mel matrix multiplication.
        with torch.amp.autocast(device_type=waveform.device.type, enabled=False):
            mel = torch.matmul(self.mel_basis, power)
        return ((mel + 0.00001).log() + 4.5) / 5.0
