import sys
import timm
import copy
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F

from model_zoo.gem import GeM
# from model_zoo.pad_conv import replace_conv2d_same
from util.torch import load_model_weights

from transformers import AutoConfig
from model_zoo.seq import DebertaV2Output, SequenceLayer
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder

warnings.simplefilter(action="ignore", category=UserWarning)


def define_model(
    name,
    num_classes=2,
    num_classes_aux=0,
    n_channels=1,
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
        head_3d="",
        n_frames=1
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
        self.head_3d = head_3d

        self.global_pool = GeM(p_trainable=False)
        self.dropout = nn.Dropout(drop_rate) if drop_rate else nn.Identity()
        
        if head_3d == "lstm":
            self.lstm = nn.LSTM(self.nb_ft, self.nb_ft // 4, batch_first=True, bidirectional=True)

        elif head_3d == "transfo":
            name = "microsoft/deberta-v3-base"
            config = AutoConfig.from_pretrained(name, output_hidden_states=True)
            config.hidden_size = self.nb_ft
            config.intermediate_size = self.nb_ft
            config.output_size = self.nb_ft // 2
            config.num_hidden_layers = 1
            config.num_attention_heads = 8
            config.attention_probs_dropout_prob = 0.1
            config.hidden_dropout_prob = 0.1
            config.hidden_act = nn.Mish()
            config.max_relative_positions = n_frames
            config.position_buckets = n_frames
            config.skip_output = True
            self.transfo = DebertaV2Encoder(config)
            self.transfo.layer[0].output = DebertaV2Output(config)
    
        elif head_3d == "cnn":
            num_layers = 1 # if n_frames == 3 else 2
            temporal_k_size = n_frames if num_layers == 1 else 3
            
            self.cnn = []
            for m in list(self.encoder.modules())[::-1]:
                if num_layers == 0:
                    break

                if "ConvNeXtBlock" in str(type(m)):
                    orig_conv = m.conv_dw
                    
                    sequence_conv = SequenceLayer(
                        n_frames,
                        orig_conv.in_channels,
                        orig_conv.out_channels,
                        orig_conv.kernel_size,
                        orig_conv.padding,
                        orig_conv.groups,
                        bias=True,
                        temporal_k_size=temporal_k_size,
                    )
                    for j in range(temporal_k_size):
                        sequence_conv.conv.weight.data[:, :, j, :, :].copy_(
                            orig_conv.weight.data / temporal_k_size   # ???
                        )
                    sequence_conv.conv.bias.data.copy_(orig_conv.bias.data)

                    m.conv_dw = sequence_conv
                    num_layers -= 1
#                     print("Replace", orig_conv, "with", sequence_conv)
                    self.cnn.append(copy.deepcopy(m))
                    m.conv_dw = nn.Identity()
                    m.norm = nn.Identity()
                    m.mlp = nn.Identity()
                    m.conv_pw = nn.Identity()
            
            self.cnn = nn.Sequential(*self.cnn[::-1])
            self.cnn.seq_length = n_frames
            
        self.logits = nn.Linear(self.nb_ft, num_classes)
        if self.num_classes_aux:
            self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

        self._update_num_channels()

    def _update_num_channels(self):
        if self.n_channels != 3:
            if "convnext" in self.encoder.name:
                conv = self.encoder.stem[0]
                new_conv = nn.Conv2d(self.n_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding)

                new_conv_w = new_conv.weight.clone().detach()
                new_conv_w[:, :3] = conv.weight.clone().detach()
                new_conv.weight = torch.nn.Parameter(new_conv_w, requires_grad=True)

                new_conv_b = conv.bias.clone().detach()
                new_conv.bias = torch.nn.Parameter(new_conv_b, requires_grad=True)
                
                self.encoder.stem[0] = new_conv
            else:
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

    def increase_stride(self):
        if "efficient" in self.encoder.name:
            self.encoder.conv_stem.stride = (4, 4)
        elif "nfnet" in self.encoder.name:
            self.encoder.stem.conv1.stride = (4, 4)
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

        if self.head_3d != "cnn":
            if self.use_gem:
                fts = self.global_pool(fts)[:, :, 0, 0]
            else:
                while len(fts.size()) > 2:
                    fts = fts.mean(-1)
                    
            fts = self.dropout(fts)
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
    
    def forward_head_3d(self, x):
        if self.head_3d == "avg":
            return x.mean(1)
        elif self.head_3d == "max":
            return x.amax(1)

        elif self.head_3d == "lstm":
            x, _ = self.lstm(x)
            mean = x.mean(1)
            max_ = x.amax(1)
            x = torch.cat([mean, max_], -1)
        elif self.head_3d == "transfo":
            x = self.transfo(
                x,
                torch.ones((x.size(0), x.size(1))).to(x.device).bool()
            ).last_hidden_state
            mean = x.mean(1)
            max_ = x.amax(1)
            x = torch.cat([mean, max_], -1)
        elif self.head_3d == "cnn":
            if len(x.shape) == 5:  # bs x n_frames x c x h x w -> bs * n_frames x c x h x w
                x = x.flatten(0, 1)

            x = self.cnn(x)

            if self.use_gem:
                x = self.global_pool(x)[:, :, 0, 0]
            else:
                while len(x.size()) > 2:
                    x = x.mean(-1)
                    
            x = self.dropout(x)  # - NOT USED BUT SHOULD PROBABLY BE

            x = x.view(x.size(0) // self.cnn.seq_length, self.cnn.seq_length, -1).mean(1)

        return x
    
    def forward_from_features(self, fts):
        """
        fts = bs x b_frames x n_fts
        """
        fts = self.dropout(fts)
        
        if self.head_3d:
            fts = self.forward_head_3d(fts)

        logits, logits_aux = self.get_logits(fts)
        return logits
    
    def set_mode(self, mode):
        if mode == "ft":
            self.forward = lambda x: self.forward_from_features(x)
        elif mode == "img":
            self.forward = lambda x: self.extract_features(x)
        else:
            raise NotImplementedError
        

    def forward(self, x, return_fts=False):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_frames x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        if self.head_3d:
            bs, n_frames, c, h, w = x.size()
            x = x.view(bs * n_frames, c, h, w)
            
        fts = self.extract_features(x)

        if self.head_3d:
            if self.head_3d != "cnn":
                fts = fts.view(bs, n_frames, -1)
            fts = self.forward_head_3d(fts)

        logits, logits_aux = self.get_logits(fts)

        if return_fts:
            return logits, fts
        return logits, logits_aux
