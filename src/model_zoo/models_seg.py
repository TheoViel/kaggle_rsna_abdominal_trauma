import torch
import torch.nn as nn
import segmentation_models_pytorch

from timm.models.layers import Conv2dSame
from model_zoo.conv3d_same import Conv3dSame


DECODERS = ["Unet"]


def define_model(
    decoder_name,
    encoder_name,
    num_classes=1,
    num_classes_aux=1,
    increase_stride=False,
    encoder_weights="imagenet",
    pretrained=True,
    use_cls=False,
    n_channels=3,
    pretrained_weights=None,
    use_3d=False,
    verbose=0,
):
    """
    Define a segmentation model.

    Args:
        decoder_name (str): Name of the decoder architecture.
        encoder_name (str): Name of the encoder architecture.
        num_classes (int, optional): Number of primary classes. Default is 1.
        num_classes_aux (int, optional): Number of auxiliary classes. Default is 1.
        increase_stride (bool, optional): Flag to increase the stride. Default is False.
        encoder_weights (str, optional): Encoder weights source. Default is "imagenet".
        pretrained (bool, optional): Flag to use pretrained weights. Default is True.
        use_cls (bool, optional): Flag to use auxiliary classifiers. Default is False.
        n_channels (int, optional): Number of input channels. Default is 3.
        pretrained_weights (str, optional): Path to pretrained model weights. Default is None.
        use_3d (bool, optional): Flag to use a 3D model. Default is False.
        verbose (int, optional): Verbosity level. Default is 0.

    Returns:
        SegWrapper: Segmentation model with the specified configurations.

    Raises:
        AssertionError: If the decoder name is not supported.
    """
    assert decoder_name in DECODERS, "Decoder name not supported"

    decoder = getattr(segmentation_models_pytorch, decoder_name)
    model = decoder(
        encoder_name="tu-" + encoder_name,
        encoder_weights=encoder_weights if pretrained else None,
        in_channels=n_channels,
        classes=num_classes,
        aux_params={"dropout": 0.0, "classes": num_classes_aux} if use_cls else None,
    )
    model.decoder_channels = (256, 128, 64, 32, 16)
    model.num_classes = num_classes

    model = SegWrapper(model)

    if increase_stride:
        model.increase_stride()

    if use_3d:
        model = convert_3d(model)

    if pretrained_weights is not None:
        if verbose:
            print(f'\n-> Loading weights from "{pretrained_weights}"\n')
        state_dict = torch.load(pretrained_weights)
        model.load_state_dict(state_dict, strict=True)

    return model


class SegWrapper(nn.Module):
    """
    A wrapper class for segmentation models.

    Attributes:
        model (nn.Module): The underlying segmentation model.
        num_classes (int): The number of classes in the model.
    """
    def __init__(self, model):
        """
        Constructor.

        Args:
            model (nn.Module): The underlying segmentation model.
        """
        super().__init__()

        self.model = model
        self.num_classes = model.num_classes

    def increase_stride(self):
        """
        Increase the stride of the first layer of the encoder.
        """
        try:
            self.model.encoder.model.conv1[3].stride = (2, 2)
        except Exception:
            self.model.encoder.model.conv1.stride = (4, 4)

        self.model.segmentation_head = (
            segmentation_models_pytorch.base.SegmentationHead(
                in_channels=self.model.decoder_channels[-1],
                upsampling=2,
                out_channels=self.model.num_classes,
                activation=None,
                kernel_size=3,
            )
        )

    def forward(self, x):
        """
        Forward pass of the segmentation model.

        Args:
            x (torch.Tensor): Input data as a tensor.

        Returns:
            torch.Tensor: Segmentation masks.
            torch.Tensor: Labels.
        """
        features = self.model.encoder(x)

        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)

        if self.model.classification_head is not None:
            labels = self.model.classification_head(features[-1])
        else:
            labels = torch.zeros(x.size(0), 1).to(x.device)

        return masks, labels


def convert_3d(module):
    """
    Convert a 2D module to its 3D counterpart.
    Adapted from: https://www.kaggle.com/code/haqishen/rsna-2022-1st-place-solution-train-stage1

    Args:
        module (torch.nn.Module): The 2D module to be converted to 3D.

    Returns:
        torch.nn.Module: The 3D equivalent of the input 2D module.
    """
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        )

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        )

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.UpsamplingBilinear2d):
        module_output = torch.nn.Upsample(
            scale_factor=module.scale_factor, mode="trilinear"
        )

    for name, child in module.named_children():
        module_output.add_module(name, convert_3d(child))
    del module

    return module_output
