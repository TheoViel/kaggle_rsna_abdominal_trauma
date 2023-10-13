import torch
import torch.nn as nn
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout


class DebertaV2Output(nn.Module):
    """
    Modified DebertaV2Output Layer. We changed the position of the skip connection,
    to allow for output_size != intermediate_size.

    Attributes:
        dense (Linear): The linear transformation layer.
        LayerNorm (LayerNorm): The layer normalization layer.
        dropout (StableDropout): The dropout layer.
        config (DebertaV2Config): The model configuration class instance.

    Methods:
        __init__(self, config): Initializes a DebertaV2Output instance with the specified config.
        forward(self, hidden_states, input_tensor): Performs the forward pass.
    """
    def __init__(self, config):
        """
        Constructor.

        Args:
            config (DebertaV2Config): The model configuration class instance.
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.output_size)
        self.LayerNorm = nn.LayerNorm(config.output_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        """
        Performs the forward pass.

        Args:
            hidden_states (Tensor): The hidden states from the previous layer.
            input_tensor (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if self.config.skip_output:
            hidden_states = self.dense(hidden_states + input_tensor)
        else:
            hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SequenceLayer(nn.Module):
    """
    From https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place/
    """
    def __init__(
        self, seq_length, in_channels, out_channels, kernel_size, padding, groups, bias, temporal_k_size=3
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(temporal_k_size, kernel_size[0], kernel_size[1]),
            padding=(temporal_k_size // 2, padding[0], padding[1]),
            groups=groups,
            bias=bias,
        )
        self.seq_length = seq_length

    def forward(self, x):        
        bs, c, h, w = x.shape
        x = x.view(bs // self.seq_length, self.seq_length, c, h, w)
        x = self.conv(x.transpose(1, 2).contiguous()).transpose(2, 1).contiguous()
        x = x.flatten(0, 1)
        return x

    
class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim=None):
        super().__init__()
 
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        if self.attention_dim is None:
            self.attention_dim = self.hidden_dim
        # W * x + b
        self.proj_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        # v.T
        self.proj_v = nn.Linear(self.attention_dim, 1, bias=False)
 
    def forward(self, x):
        """
        :param x: seq_len, batch_size, hidden_dim
        :return: batch_size * seq_len, batch_size * hidden_dim
        """
        # print(f"x shape:{x.shape}")
        batch_size, seq_len, _ = x.size()
        # flat_inputs = x.reshape(-1, self.hidden_dim) # (batch_size*seq_len, hidden_dim)
        # print(f"flat_inputs shape:{flat_inputs.shape}")
        
        H = torch.tanh(self.proj_w(x)) # (batch_size, seq_len, hidden_dim)
        # print(f"H shape:{H.shape}")
        
        att_scores = torch.softmax(self.proj_v(H),axis=1) # (batch_size, seq_len)
        # print(f"att_scores shape:{att_scores.shape}")
        
        attn_x = (x * att_scores).sum(1) # (batch_size, hidden_dim)
        # print(f"attn_x shape:{attn_x.shape}")
        return attn_x
