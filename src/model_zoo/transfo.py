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
    