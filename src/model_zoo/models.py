import sys
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_zoo.gem import GeM
# from model_zoo.pad_conv import replace_conv2d_same

from util.torch import load_model_weights

# sys.path.append("../Next-ViT/classification")
# import nextvit

# WEIGHTS = {
#     "nextvit_small": "../input/nextvit_small_in1k6m_384.pth",
#     "nextvit_base": "../input/nextvit_base_in1k6m_384.pth",
#     "nextvit_large": "../input/nextvit_large_in1k6m_384.pth",
# }


def define_model(
    name,
    num_classes=2,
    num_classes_aux=0,
    n_channels=1,
    pretrained_weights="",
    pretrained=True,
    reduce_stride=False,
    drop_rate=0,
    drop_path_rate=0,
    use_gem=False,
    verbose=1,
    replace_pad_conv=False,
):
    """
    Loads a pretrained model & builds the architecture.
    Supports timm models.
    TODO

    Args:
        name (str): Model name
        num_classes (int, optional): Number of classes. Defaults to 1.
        num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
        n_channels (int, optional): Number of image channels. Defaults to 3.
        pretrained_weights (str, optional): Path to pretrained encoder weights. Defaults to ''.
        pretrained (bool, optional): Whether to load timm pretrained weights.
        reduce_stride (bool, optional): Whether to reduce first layer stride. Defaults to False.

    Returns:
        torch model -- Pretrained model.
    """

#     if "nextvit" in name:
#         encoder = getattr(nextvit, name)(pretrained=pretrained)
#         if pretrained:
#             encoder.load_state_dict(torch.load(WEIGHTS[name])["model"], strict=True)
#         encoder.num_features = encoder.proj_head[0].in_features
#         del encoder.proj_head, encoder.avgpool
    if drop_path_rate > 0:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            global_pool='',
        )
    else:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
    encoder.name = name

    model = ClsModel(
        encoder,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_channels=n_channels,
        drop_rate=drop_rate,
        use_gem=use_gem,
    )

    if pretrained_weights:
        model = load_model_weights(
            model, pretrained_weights, verbose=verbose, strict=False
        )

    if reduce_stride:
        model.reduce_stride()
        
    if replace_pad_conv and "efficient" in name:
        if verbose:
            print('Replacing Conv2dSame layers\n')
        model = replace_conv2d_same(model, verbose=0)

    return model


class ClsModel(nn.Module):
    """
    Model with an attention mechanism.
    """

    def __init__(
        self,
        encoder,
        num_classes=2,
        num_classes_aux=11,
        n_channels=3,
        drop_rate=0,
        use_gem=False,
    ):
        """
        Constructor.
        TODO

        Args:
            encoder (timm model): Encoder.
            num_classes (int, optional): Number of classes. Defaults to 1.
            num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
            n_channels (int, optional): Number of image channels. Defaults to 3.
        """
        super().__init__()

        self.encoder = encoder
        self.nb_ft = encoder.num_features

        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.n_channels = n_channels
        self.use_gem = use_gem

        self.global_pool = GeM(p_trainable=False)
        self.dropout = nn.Dropout(drop_rate) if drop_rate else nn.Identity()

        self.logits = nn.Linear(self.nb_ft, num_classes)
        if self.num_classes_aux:
            self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

        self._update_num_channels()

    def _update_num_channels(self):
        if self.n_channels != 3:
            for n, m in self.encoder.named_modules():
                if n:
                    # print("Replacing", n)
                    old_conv = getattr(self.encoder, n)
                    new_conv = nn.Conv2d(
                        self.n_channels,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None,
                    )
                    setattr(self.encoder, n, new_conv)
                    break

    def reduce_stride(self):
        if "efficient" in self.encoder.name:
            self.encoder.conv_stem.stride = (1, 1)
        elif "nfnet" in self.encoder.name:
            self.encoder.stem.conv1.stride = (1, 1)
        else:
            raise NotImplementedError

    def extract_features(self, x):
        """
        Extract features function.

        Args:
            x (torch tensor [batch_size x 3 x w x h]): Input batch.

        Returns:
            torch tensor [batch_size x num_features]: Features.
        """
        fts = self.encoder(x)

        if self.use_gem:
            fts = self.global_pool(fts)[:, :, 0, 0]
        else:
            while len(fts.size()) > 2:
                fts = fts.mean(-1)

        return fts

    def get_logits(self, fts):
        """
        Computes logits.

        Args:
            fts (torch tensor [batch_size x num_features]): Features.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        logits = self.logits(fts)

        if self.num_classes_aux:
            logits_aux = self.logits_aux(fts)
        else:
            logits_aux = torch.zeros((fts.size(0)))

        return logits, logits_aux

    def forward(self, x, return_fts=False):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_frames x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        fts = self.extract_features(x)

        fts = self.dropout(fts)

        logits, logits_aux = self.get_logits(fts)

        if return_fts:
            return logits, fts
        return logits, logits_aux
