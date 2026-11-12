"""Verbatim TorchAudio 2.7.1 sinc resampling closure, implemented only with Torch.

Preserves the original numerical algorithm and ordering without its binary runtime.
Source: pytorch/audio 95c61b4168fc5133be8dd8c1337d929d066ae6cf,
functional/functional.py; SHA256 73caabde698f5cb8b8d0de0d08b7291ef41e508ba836d6ecce4b23b6bb82d798.
Copyright (c) 2017 Facebook Inc. (Soumith Chintala); BSD-2-Clause.
Full notice and conditions: licenses/TORCHAUDIO_LICENSE.txt.
Reads: Torch tensors, source/target rates and native filter parameters.
"""

import math
import warnings
from typing import Optional

import torch
from torch import Tensor

_CPU = torch.device("cpu")


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
    device: torch.device = _CPU,
    dtype: Optional[torch.dtype] = None,
):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception(
            "Frequencies must be of integer type to ensure quality resampling computation. "
            "To work around this, manually convert both frequencies to integer values "
            "that maintain their resampling rate ratio before passing them into the function. "
            "Example: To downsample a 44100 hz waveform by a factor of 8, use "
            "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
            "For more information, please refer to https://github.com/pytorch/audio/issues/1487."
        )

    if resampling_method in ["sinc_interpolation", "kaiser_window"]:
        method_map = {
            "sinc_interpolation": "sinc_interp_hann",
            "kaiser_window": "sinc_interp_kaiser",
        }
        warnings.warn(
            f'"{resampling_method}" resampling method name is being deprecated and replaced by '
            f'"{method_map[resampling_method]}" in the next release. '
            "The default behavior remains unchanged.",
            stacklevel=3,
        )
    elif resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= rolloff

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq

    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    # we do not use built in torch windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: Tensor,
    width: int,
):
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled


def resample(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
) -> Tensor:
    r"""Resamples the waveform at the new frequency using bandlimited interpolation. :cite:`RESAMPLE`.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        ``transforms.Resample`` precomputes and reuses the resampling kernel, so using it will result in
        more efficient computation if resampling multiple waveforms with the same resampling parameters.

    Args:
        waveform (Tensor): The input signal of dimension `(..., time)`
        orig_freq (int): The original frequency of the signal
        new_freq (int): The desired frequency
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``"sinc_interp_hann"``, ``"sinc_interp_kaiser"``] (Default: ``"sinc_interp_hann"``)
        beta (float or None, optional): The shape parameter used for kaiser window.

    Returns:
        Tensor: The waveform at the new frequency of dimension `(..., time).`
    """

    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError("Original frequency and desired frequecy should be positive")

    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.device,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled
