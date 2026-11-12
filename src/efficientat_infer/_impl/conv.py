"""Pure Torch replacement for the TorchVision convolution wrapper used by MN.

Keeps child indices 0/1/2 and constructor defaults so release state dicts load strictly.
Reads: Torch's Conv2d, BatchNorm2d, and activation modules.
"""

from torch import nn


class ConvNormActivation(nn.Sequential):
    """The TorchVision ConvNormActivation subset exercised by EfficientAT MN."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        dilation=1,
        inplace=True,
        bias=None,
    ):
        if padding is None:
            if isinstance(kernel_size, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                padding = tuple((k - 1) // 2 * d for k, d in zip(kernel_size, dilation))
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, groups=groups, bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels
