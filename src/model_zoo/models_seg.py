# TODO : DOC

import torch
import torch.nn as nn
import segmentation_models_pytorch

# from model_zoo.unet import Unet


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
    verbose=0,
):
    """
    Define a segmentation model.

    Args:
        decoder_name (str): Name of the decoder architecture.
        encoder_name (str): Name of the encoder architecture.
        num_classes (int, optional): Number of output classes. Defaults to 1.
        encoder_weights (str, optional): Type of encoder weights to use. Defaults to "imagenet".
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        n_channels (int, optional): Number of input channels. Defaults to 3.
        pretrained_weights (str, optional): Path to pretrained weights file. Defaults to None.
        use_cls (bool, optional): Whether to use auxiliary classification head. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        nn.Module: Defined segmentation model.
    """
    assert decoder_name in DECODERS, "Decoder name not supported"

    decoder = getattr(segmentation_models_pytorch, decoder_name)
    model = decoder(
        encoder_name="tu-" + encoder_name,
        encoder_weights=encoder_weights if pretrained else None,
        in_channels=n_channels,
        classes=num_classes,
        aux_params={"dropout": 0., "classes": num_classes_aux} if use_cls else None,
#         decoder_channels=(128, 64, 32, 16, 8),
    )
#     model.decoder_channels = (128, 64, 32, 16, 8)
    model.decoder_channels = (256, 128, 64, 32, 16)
    model.num_classes = num_classes

    model = SegWrapper(model)
    
    if increase_stride:
        model.increase_stride()

    if pretrained_weights is not None:
        if verbose:
            print(f'\n-> Loading weights from "{pretrained_weights}"\n')
        state_dict = torch.load(pretrained_weights)
#         del (
#             state_dict['model.segmentation_head.0.weight'],
#             state_dict['model.segmentation_head.0.bias'],
#         )
        model.load_state_dict(state_dict, strict=False)
        
    return model


class SegWrapper(nn.Module):
    """
    Wrapper module for segmentation models with optional auxiliary classification and transformation layers.

    This class wraps a segmentation model and adds optional auxiliary classification or 3D layers,
    such as LSTM, CNN, or transformer.
    It supports reducing stride for certain encoders and decoding architectures.

    Attributes:
        model (nn.Module): The wrapped segmentation model.
        num_classes (int): Number of output classes from the segmentation model.
        use_cls (bool): Whether auxiliary classification is enabled.
        use_lstm (bool): Whether LSTM is enabled.
        use_cnn (bool): Whether CNN is enabled.
        use_transfo (bool): Whether transformer is enabled.
        two_layers (bool): Whether two temporal mixing layers are used.

    Methods:
        forward(x):
            Forward pass through the wrapped model with optional auxiliary layers.

    """
    def __init__(self, model):
        """
        Constructor.

        Args:
            model (nn.Module): The segmentation model to wrap.
        """
        super().__init__()

        self.model = model
        self.num_classes = model.num_classes
        
    def increase_stride(self):
        try:
#             model.encoder.model.conv1[0].stride = (2, 2)
            self.model.encoder.model.conv1[3].stride = (2, 2)
            self.model.encoder.model.conv1[6].stride = (2, 2)
        except:
            self.model.encoder.model.conv1.stride = (4, 4)

        self.model.segmentation_head = segmentation_models_pytorch.base.SegmentationHead(
            in_channels=self.model.decoder_channels[-1],
            upsampling=4,
            out_channels=self.model.num_classes,
            activation=None,
            kernel_size=3,
        )       

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_frames x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        features = self.model.encoder(x)

        decoder_output = self.model.decoder(*features)

        masks = self.model.segmentation_head(decoder_output)
        
        if self.model.classification_head is not None:
            labels = self.model.classification_head(features[-1])
        else:
            labels = torch.zeros(x.size(0), 1).to(x.device)

        return masks, labels
