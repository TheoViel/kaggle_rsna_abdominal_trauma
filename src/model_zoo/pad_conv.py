import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from timm.models.layers.conv2d_same import Conv2dSame


class MyConv2dSame_(nn.Conv2d):
    """
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(MyConv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def forward(self, x):
        x = F.pad(x, [0, 1, 0, 1], value=0.0)
        return F.conv2d(
            x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups
        )


class MyConv2dSame(nn.Conv2d):
    """
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(MyConv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def forward(self, x):
        return F.conv2d(
            x, self.weight, self.bias, self.stride, (1, 1), self.dilation, self.groups
        )


def replace_conv2d_same(model, verbose=0):
    new_model = deepcopy(model)

    for n, m in [
        ("conv_stem", model.encoder.conv_stem),
        ("conv_head", model.encoder.conv_head),
    ]:
        if isinstance(m, Conv2dSame):
            if verbose:
                print(f"Replacing {n}")

            new_conv = MyConv2dSame(
                m.in_channels,
                m.out_channels,
                m.kernel_size,
                stride=m.stride,
                dilation=m.dilation,
                groups=m.groups,
                bias=m.bias is not None,
            )
            new_conv.weight = torch.nn.Parameter(torch.clone(m.weight))
            if m.bias is not None:
                new_conv.bias = torch.nn.Parameter(torch.clone(m.bias))

            setattr(new_model.encoder, n, new_conv)

    for block_idx, block in enumerate(model.encoder.blocks):
        for layer_idx, layer in enumerate(deepcopy(block)):
            for n, m in layer.named_modules():
                if isinstance(m, Conv2dSame):
                    new_conv = MyConv2dSame(
                        m.in_channels,
                        m.out_channels,
                        m.kernel_size,
                        stride=m.stride,
                        dilation=m.dilation,
                        groups=m.groups,
                        bias=m.bias is not None,
                    )
                    #                     print(m.bias)
                    new_conv.weight = torch.nn.Parameter(torch.clone(m.weight))
                    if m.bias is not None:
                        new_conv.bias = torch.nn.Parameter(torch.clone(m.bias))

                    if verbose:
                        print(f"Replacing block {block_idx} - layer {layer_idx} -", n)

                    if "se." in n:
                        setattr(
                            new_model.encoder.blocks[block_idx][layer_idx].se,
                            n[3:],
                            new_conv,
                        )
                    else:
                        setattr(
                            new_model.encoder.blocks[block_idx][layer_idx], n, new_conv
                        )

    return new_model
