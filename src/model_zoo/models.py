import timm
import torch
import warnings
import torch.nn as nn

from model_zoo.layers import GeM, Attention
from util.torch import load_model_weights

warnings.simplefilter(action="ignore", category=UserWarning)


def define_model(
    name,
    num_classes=2,
    num_classes_aux=0,
    n_channels=3,
    pretrained_weights="",
    pretrained=True,
    reduce_stride=False,
    increase_stride=False,
    drop_rate=0,
    drop_path_rate=0,
    use_gem=False,
    head_3d="",
    n_frames=1,
    verbose=1,
    replace_pad_conv=False,
):
    """
    Loads a pretrained model & builds the architecture.
    Supports timm models.

    Args:
        name (str): Name of the model architecture.
        num_classes (int, optional): Number of main output classes. Defaults to 2.
        num_classes_aux (int, optional): Number of auxiliary output classes. Defaults to 0.
        n_channels (int, optional): Number of input channels. Defaults to 3.
        pretrained_weights (str, optional): Path to pre-trained weights. Defaults to "".
        pretrained (bool, optional): Whether to use pre-trained weights. Defaults to True.
        reduce_stride (bool, optional): Whether to reduce the model's stride. Defaults to False.
        increase_stride (bool, optional): Whether to increase the model's stride. Defaults to False.
        drop_rate (float, optional): Dropout rate. Defaults to 0.
        drop_path_rate (float, optional): Drop path rate. Defaults to 0.
        use_gem (bool, optional): Whether to use GeM pooling. Defaults to False.
        head_3d (str, optional): 3D head type. Defaults to "".
        n_frames (int, optional): Number of frames. Defaults to 1.
        verbose (int, optional): Verbosity level. Defaults to 1.
        replace_pad_conv (bool, optional): Whether to replace padding convolution. Defaults to False.

    Returns:
        ClsModel: The defined model.
    """
    if drop_path_rate > 0 and "coat" not in name:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            global_pool="avg" if "coat" in name else "",
        )
    else:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg" if "coat" in name else "",
        )
    encoder.name = name

    model = ClsModel(
        encoder,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_channels=n_channels,
        drop_rate=drop_rate,
        use_gem=use_gem,
        head_3d=head_3d,
        n_frames=n_frames,
    )

    if pretrained_weights:
        model = load_model_weights(
            model, pretrained_weights, verbose=verbose, strict=False
        )

    if reduce_stride:
        model.reduce_stride()
    if increase_stride:
        model.increase_stride()

    return model


class ClsModel(nn.Module):
    """
    PyTorch model for image classification.

    Attributes:
        encoder: The feature encoder.
        num_classes (int): The number of primary classes.
        num_classes_aux (int): The number of auxiliary classes.
        n_channels (int): The number of input channels.
        drop_rate (float): Dropout rate.
        use_gem (bool): Flag to use Generalized Mean Pooling (GeM).
        head_3d (str): The 3D head type.
        n_frames (int): The number of frames.
    """
    def __init__(
        self,
        encoder,
        num_classes=2,
        num_classes_aux=11,
        n_channels=3,
        drop_rate=0,
        use_gem=False,
        head_3d="",
        n_frames=1,
    ):
        """
        Constructor for the classification model.

        Args:
            encoder: The feature encoder.
            num_classes (int): The number of primary classes.
            num_classes_aux (int): The number of auxiliary classes.
            n_channels (int): The number of input channels.
            drop_rate (float): Dropout rate.
            use_gem (bool): Flag to use Generalized Mean Pooling (GeM).
            head_3d (str): The 3D head type.
            n_frames (int): The number of frames.
        """
        super().__init__()

        self.encoder = encoder
        self.nb_ft = encoder.num_features

        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.n_channels = n_channels
        self.use_gem = use_gem
        self.head_3d = head_3d

        self.global_pool = GeM(p_trainable=False)
        self.dropout = nn.Dropout(drop_rate) if drop_rate else nn.Identity()

        if head_3d == "lstm":
            self.lstm = nn.LSTM(
                self.nb_ft, self.nb_ft // 4, batch_first=True, bidirectional=True
            )
        elif head_3d == "lstm_att":
            self.lstm = nn.LSTM(
                self.nb_ft, self.nb_ft // 2, batch_first=True, bidirectional=True
            )
            self.att = Attention(self.nb_ft, self.nb_ft)
        elif head_3d == "transfo":
            self.transfo = nn.TransformerEncoderLayer(
                self.nb_ft,
                8,
                dim_feedforward=self.nb_ft * 2,
                dropout=0.1,
                activation=nn.Mish(),
                batch_first=True,
            )

        self.logits = nn.Linear(self.nb_ft, num_classes)
        if self.num_classes_aux:
            self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

        self._update_num_channels()

    def _update_num_channels(self):
        """
        Update the number of input channels for the encoder.
        """
        if self.n_channels != 3:
            if "convnext" in self.encoder.name:
                conv = self.encoder.stem[0]
            elif "coat_lite" in self.encoder.name:
                conv = self.encoder.patch_embed1.proj
            elif "coatnet" in self.encoder.name:
                conv = self.encoder.stem.conv1

            new_conv = nn.Conv2d(
                self.n_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
            )

            new_conv_w = new_conv.weight.clone().detach()
            new_conv_w[:, :3] = conv.weight.clone().detach()
            new_conv.weight = torch.nn.Parameter(new_conv_w, requires_grad=True)

            if conv.bias is not None:
                new_conv_b = conv.bias.clone().detach()
                new_conv.bias = torch.nn.Parameter(new_conv_b, requires_grad=True)

            if "convnext" in self.encoder.name:
                self.encoder.stem[0] = new_conv
            elif "coat_lite" in self.encoder.name:
                self.encoder.patch_embed1.proj = new_conv
            elif "coatnet" in self.encoder.name:
                self.encoder.stem.conv1 = new_conv

    def reduce_stride(self):
        """
        Reduce the stride of the first layer of the encoder.
        """
        if "efficient" in self.encoder.name:
            self.encoder.conv_stem.stride = (1, 1)
        elif "nfnet" in self.encoder.name:
            self.encoder.stem.conv1.stride = (1, 1)
        else:
            raise NotImplementedError

    def increase_stride(self):
        """
        Increase the stride of the first layer of the encoder.
        """
        if "efficient" in self.encoder.name:
            self.encoder.conv_stem.stride = (4, 4)
        elif "nfnet" in self.encoder.name:
            self.encoder.stem.conv1.stride = (4, 4)
        else:
            raise NotImplementedError

    def extract_features(self, x):
        """
        Extract features from input images.

        Args:
            x (torch.Tensor): Input images of shape [batch_size x n_channels x H x W].

        Returns:
            torch.Tensor: Extracted features of shape [batch_size x num_features].
        """
        fts = self.encoder(x)

        if "swin" in self.encoder.name:
            fts = fts.transpose(2, 3).transpose(1, 2)

        if self.use_gem and len(fts.size()) == 4:
            fts = self.global_pool(fts)[:, :, 0, 0]
        else:
            while len(fts.size()) > 2:
                fts = fts.mean(-1)

        fts = self.dropout(fts)

        return fts

    def get_logits(self, fts):
        """
        Compute logits for the primary and auxiliary classes.

        Args:
            fts (torch.Tensor): Features of shape [batch_size x num_features].

        Returns:
            torch.Tensor: Logits for the primary classes of shape [batch_size x num_classes].
            torch.Tensor: Logits for the auxiliary classes of shape [batch_size x num_classes_aux].
        """
        logits = self.logits(fts)

        if self.num_classes_aux:
            logits_aux = self.logits_aux(fts)
        else:
            logits_aux = torch.zeros((fts.size(0)))

        return logits, logits_aux

    def forward_head_3d(self, x):
        """
        Forward function for the 3D head.

        Args:
            x (torch.Tensor): Input features for the 3D head of shape [batch_size x n_frames x num_features].

        Returns:
            torch.Tensor: Result of the 3D head.
        """
        if self.head_3d == "avg":
            return x.mean(1)
        elif self.head_3d == "max":
            return x.amax(1)

        elif self.head_3d == "lstm":
            x, _ = self.lstm(x)
            mean = x.mean(1)
            max_ = x.amax(1)
            x = torch.cat([mean, max_], -1)
        elif self.head_3d == "lstm_att":
            x, _ = self.lstm(x)
            x = self.att(x)

        elif self.head_3d == "transfo":
            x = self.transfo(x).mean(1)

        return x

    def forward_from_features(self, fts):
        """
        Forward function using pre-extracted features.

        Args:
            fts (torch.Tensor): Features of shape [batch_size x n_frames x num_features].

        Returns:
            torch.Tensor: Logits for the primary classes of shape [batch_size x num_classes].
        """
        fts = self.dropout(fts)

        if self.head_3d:
            fts = self.forward_head_3d(fts)

        logits, logits_aux = self.get_logits(fts)

        return logits

    def set_mode(self, mode):
        """
        Set the forward mode (features or images).

        Args:
            mode (str): The mode to set, "ft" for features or "img" for images.
        """
        if mode == "ft":
            self.forward = lambda x: self.forward_from_features(x)
        elif mode == "img":
            self.forward = lambda x: self.extract_features(x)
        else:
            raise NotImplementedError

    def forward(self, x, return_fts=False):
        """
        Forward function for the model.

        Args:
            x (torch.Tensor): Input images of shape [batch_size (x n_frames) x n_channels x H x W].
            return_fts (bool): Flag to return features.

        Returns:
            torch.Tensor: Logits for the primary classes of shape [batch_size x num_classes].
            torch.Tensor: Logits for the auxiliary classes of shape [batch_size x num_classes_aux].
        """
        if self.head_3d:
            bs, n_frames, c, h, w = x.size()
            x = x.view(bs * n_frames, c, h, w)

        fts = self.extract_features(x)

        if self.head_3d:
            fts = fts.view(bs, n_frames, -1)
            fts = self.forward_head_3d(fts)

        logits, logits_aux = self.get_logits(fts)

        if return_fts:
            return logits, fts
        return logits, logits_aux
