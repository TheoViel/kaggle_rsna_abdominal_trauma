import torch
import torch.nn as nn


def define_model(
    name="rnn_att",
    ft_dim=2048,
    layer_dim=64,
    n_layers=1,
    dense_dim=256,
    p=0.1,
    num_classes=2,
    num_classes_aux=0,
    n_fts=0,
):
    """
    Define the level 2 model.

    Args:
        name (str): The name of the model to define. Default is "rnn_att".
        ft_dim (int): Dimension of input features. Default is 2048.
        layer_dim (int): Dimension of LSTM layers. Default is 64.
        n_layers (int): Number of LSTM layers. Default is 1.
        dense_dim (int): Dimension of the dense layer. Default is 256.
        p (float): Dropout probability. Default is 0.1.
        num_classes (int): Number of main classes. Default is 2.
        num_classes_aux (int): Number of auxiliary classes. Default is 0.
        n_fts (int): Number of features to use. Default is 0.

    Returns:
        nn.Module: The defined model.
    """
    if name == "rnn_att":
        pass
#         return RNNAttModel(
#             ft_dim=ft_dim,
#             lstm_dim=layer_dim,
#             n_lstm=n_layers,
#             dense_dim=dense_dim,
#             p=p,
#             num_classes=num_classes,
#             num_classes_aux=num_classes_aux,
#             n_fts=n_fts,
#         )
    elif name == "rnn_spleen":
        return RNNSpleenModel(
            ft_dim=ft_dim,
            lstm_dim=layer_dim,
            n_lstm=n_layers,
            dense_dim=dense_dim,
            p=p,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_fts=n_fts,
        )
    elif name == "rnn_bowel":
        return RNNBowelModel(
            ft_dim=ft_dim,
            lstm_dim=layer_dim,
            n_lstm=n_layers,
            dense_dim=dense_dim,
            p=p,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_fts=n_fts,
        )
    else:
        raise NotImplementedError


class RNNBowelModel(nn.Module):
    """
    Recurrent Neural Network with attention.

    Attributes:
        ft_dim (int): The dimension of input features.
        lstm_dim (int): The dimension of the LSTM layer.
        n_lstm (int): The number of LSTM layers.
        dense_dim (int): The dimension of the dense layer.
        p (float): Dropout probability.
        num_classes (int): The number of primary target classes.
        num_classes_aux (int): The number of auxiliary target classes.
        n_fts (int): The number of additional features.
    """
    def __init__(
        self,
        ft_dim=64,
        lstm_dim=64,
        n_lstm=1,
        dense_dim=64,
        p=0.1,
        num_classes=8,
        num_classes_aux=0,
        n_fts=0,
    ):
        """
        Constructor.

        Args:
            ft_dim (int): The dimension of input features. Defaults to 64.
            lstm_dim (int): The dimension of the LSTM layer. Defaults to 64.
            n_lstm (int): The number of LSTM layers. Defaults to 1.
            dense_dim (int): The dimension of the dense layer. Defaults to 64.
            p (float): Dropout probability. Defaults to 0.1.
            num_classes (int): The number of primary target classes. Defaults to 8.
            num_classes_aux (int): The number of auxiliary target classes. Defaults to 0.
            n_fts (int): The number of additional features. Defaults to 0.

        """
        super().__init__()
        self.n_fts = n_fts
        self.n_lstm = n_lstm
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux

        self.mlp = nn.Sequential(
            nn.Linear(ft_dim, dense_dim),
            nn.Mish(),
        )

        if n_fts > 0:
            self.mlp_fts = nn.Sequential(
                nn.Linear(n_fts, dense_dim),
                nn.Dropout(p=p),
                nn.Mish(),
            )
            n_fts = n_fts + dense_dim

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.logits_bowel = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 4, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 1),
        )
        self.logits_extrav = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 4, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 1),
        )

        if num_classes_aux:
            raise NotImplementedError

    def attention_pooling(self, x, w):
        """
        Apply attention pooling to input features.

        Args:
            x (torch.Tensor): Input feature tensor.
            w (torch.Tensor): Attention weights.

        Returns:
            torch.Tensor: The pooled result.
        """
        return (x * w).sum(1) / (w.sum(1) + 1e-6), (x * w).amax(1)

    def forward(self, x, ft=None):
        """
        Forward pass of the RNN with attention model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_features).
            ft (torch.Tensor, optional): Additional features tensor. Default is None.

        Returns:
            torch.Tensor: Model outputs as logits for different classes.
            torch.Tensor: Placeholder for auxiliary outputs.
        """
        ft = None  # ft[:, 2:]
        seg = x[:, :, :5]
        x = x[:, :, 5:]
    
        x = torch.cat([
            seg[:, :, -1:],  # bowel
            seg.amax(-1, keepdims=True),  # extrav
            x[:, :, :2],  # bowel & extrav
            x[:, :, 11: 13],  # bowel & extrav * seg
        ], -1)

        features = self.mlp(x)
        features_lstm, _ = self.lstm(features)

        features = torch.cat([features, features_lstm], -1)

        bowel = seg[:, :, :1]
        att_bowel, max_bowel = self.attention_pooling(features, bowel)

        mean = features.mean(1)
        max_ = features.amax(1)
        
#         print(x.amax(1))
#         print(x.size())

        scores = x[:, :, 2:]
        scores = scores.view(x.size(0), x.size(1), -1, 2, 2)
        pooled_scores = scores.mean(2)  # avg across models
        pooled_scores = torch.cat([pooled_scores.amax(1), pooled_scores.mean(1)], 1)  # temporal pool
#         print(pooled_scores)

        pooled_scores_bowel = pooled_scores[:, :, 0]
        pooled_scores_extrav = pooled_scores[:, :, 1]
        
#         print(pooled_scores_bowel.size())
#         print(pooled_scores_extrav.size())

        logits_bowel = self.logits_bowel(
            torch.cat([att_bowel, max_bowel, pooled_scores_bowel], -1)
        )
        logits_extrav = self.logits_extrav(
            torch.cat([mean, max_, pooled_scores_extrav], -1)
        )
    
        logits_spleen = torch.zeros((x.size(0), 3), dtype=logits_bowel.dtype, device=logits_bowel.device)
        logits_kidney = torch.zeros((x.size(0), 3), dtype=logits_bowel.dtype, device=logits_bowel.device)
        logits_liver = torch.zeros((x.size(0), 3), dtype=logits_bowel.dtype, device=logits_bowel.device)

        logits = torch.cat(
            [logits_bowel, logits_extrav, logits_kidney, logits_liver, logits_spleen],
            -1,
        )

        return logits, torch.zeros((x.size(0)))


class RNNSpleenModel(nn.Module):
    """
    Recurrent Neural Network with attention.

    Attributes:
        ft_dim (int): The dimension of input features.
        lstm_dim (int): The dimension of the LSTM layer.
        n_lstm (int): The number of LSTM layers.
        dense_dim (int): The dimension of the dense layer.
        p (float): Dropout probability.
        num_classes (int): The number of primary target classes.
        num_classes_aux (int): The number of auxiliary target classes.
        n_fts (int): The number of additional features.
    """
    def __init__(
        self,
        ft_dim=64,
        lstm_dim=64,
        n_lstm=1,
        dense_dim=64,
        p=0.1,
        num_classes=8,
        num_classes_aux=0,
        n_fts=0,
    ):
        """
        Constructor.

        Args:
            ft_dim (int): The dimension of input features. Defaults to 64.
            lstm_dim (int): The dimension of the LSTM layer. Defaults to 64.
            n_lstm (int): The number of LSTM layers. Defaults to 1.
            dense_dim (int): The dimension of the dense layer. Defaults to 64.
            p (float): Dropout probability. Defaults to 0.1.
            num_classes (int): The number of primary target classes. Defaults to 8.
            num_classes_aux (int): The number of auxiliary target classes. Defaults to 0.
            n_fts (int): The number of additional features. Defaults to 0.

        """
        super().__init__()
        self.n_fts = n_fts
        self.n_lstm = n_lstm
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux

        self.mlp = nn.Sequential(
            nn.Linear(ft_dim, dense_dim),
            nn.Mish(),
        )

        if n_fts > 0:
            self.mlp_fts = nn.Sequential(
                nn.Linear(n_fts, dense_dim),
                nn.Dropout(p=p),
                nn.Mish(),
            )
            n_fts = n_fts + dense_dim

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

#         self.logits_bowel = nn.Sequential(
#             nn.Dropout(p=0),
#             nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 4, dense_dim),
#             nn.Mish(),
#             nn.Linear(dense_dim, 1),
#         )
#         self.logits_extrav = nn.Sequential(
#             nn.Dropout(p=0),
#             nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 4, dense_dim),
#             nn.Mish(),
#             nn.Linear(dense_dim, 1),
#         )
        self.logits_spleen = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 8 + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )
#         self.logits_liver = nn.Sequential(
#             nn.Dropout(p=0),
#             nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 8 + n_fts, dense_dim),
#             nn.Mish(),
#             nn.Linear(dense_dim, 3),
#         )
#         self.logits_kidney = nn.Sequential(
#             nn.Dropout(p=0),
#             nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 8 + n_fts, dense_dim),
#             nn.Mish(),
#             nn.Linear(dense_dim, 3),
#         )

        if num_classes_aux:
            raise NotImplementedError

    def attention_pooling(self, x, w):
        """
        Apply attention pooling to input features.

        Args:
            x (torch.Tensor): Input feature tensor.
            w (torch.Tensor): Attention weights.

        Returns:
            torch.Tensor: The pooled result.
        """
        return (x * w).sum(1) / (w.sum(1) + 1e-6), (x * w).amax(1)

    def forward(self, x, ft=None):
        """
        Forward pass of the RNN with attention model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_features).
            ft (torch.Tensor, optional): Additional features tensor. Default is None.

        Returns:
            torch.Tensor: Model outputs as logits for different classes.
            torch.Tensor: Placeholder for auxiliary outputs.
        """
#         x[:, :, :1] = 0  # liver seg
#         x[:, :, 2:5] = 0  # kidney & bowel seg
        
#         x[:, :, 5: 5 + 11 - 3] = 0  # model spleen
#         x[:, :, 5 + 11: - 3] = 0  # model*seg not spleen
        
#         ft[:, 0] = 0  # kidney ft
#         ft[:, 1] = 0  # liver ft
        
        ft = ft[:, 2:]
#         print(ft)
        
        x = torch.cat([
            x[:, :, 1:2],
            x[:, :, 5 + 11 - 3:5 + 11],
            x[:, :, -3:],
        ], -1)

        features = self.mlp(x)
        features_lstm, _ = self.lstm(features)

        features = torch.cat([features, features_lstm], -1)

#         kidney = x[:, :, 2:4].amax(-1, keepdims=True)
#         liver = x[:, :, :1]
        spleen = x[:, :, :1]
#         bowel = x[:, :, 4:5]

#         att_bowel, max_bowel = self.attention_pooling(features, bowel)
#         att_kidney, max_kidney = self.attention_pooling(features, kidney)
#         att_liver, max_liver = self.attention_pooling(features, liver)
        att_spleen, max_spleen = self.attention_pooling(features, spleen)

        mean = features.mean(1)
        max_ = features.amax(1)

        scores = x[:, :, 1:]
        scores = scores.view(x.size(0), x.size(1), -1, 3 * 2)
        pooled_scores = scores.mean(2).view(x.size(0), x.size(1), 2, 3)
        pooled_scores = torch.cat([pooled_scores.amax(1), pooled_scores.mean(1)], 1)


#         pooled_scores_bowel = pooled_scores[:, :, :1].flatten(1, 2)
#         pooled_scores_extrav = pooled_scores[:, :, 1:2].flatten(1, 2)
#         pooled_scores_kidney = pooled_scores[:, :, 3:5].flatten(1, 2)
#         pooled_scores_liver = pooled_scores[:, :, 6:8].flatten(1, 2)
        pooled_scores_spleen = pooled_scores[:, :, 1:].flatten(1, 2)

        if ft is None or self.n_fts == 0:
            ft_kidney = torch.empty((x.size(0), 0)).to(x.device)
            ft_liver = torch.empty((x.size(0), 0)).to(x.device)
            ft_spleen = torch.empty((x.size(0), 0)).to(x.device)
        else:
            fts = self.mlp_fts(ft.flatten(1, 2))
#             ft_kidney = torch.cat([fts, ft[:, 0]], -1)
#             ft_liver = torch.cat([fts, ft[:, 1]], -1)
            ft_spleen = torch.cat([fts, ft[:, 0]], -1)

#         logits_bowel = self.logits_bowel(
#             torch.cat([att_bowel, max_bowel, pooled_scores_bowel], -1)
#         )
#         logits_extrav = self.logits_extrav(
#             torch.cat([mean, max_, pooled_scores_extrav], -1)
#         )
#         logits_kidney = self.logits_kidney(
#             torch.cat([att_kidney, max_kidney, pooled_scores_kidney, ft_kidney], -1)
#         )
#         logits_liver = self.logits_liver(
#             torch.cat([att_liver, max_liver, pooled_scores_liver, ft_liver], -1)
#         )
#         print(att_spleen.size(), max_spleen.size(), pooled_scores_spleen.size(), ft_spleen.size())
        logits_spleen = self.logits_spleen(
            torch.cat([att_spleen, max_spleen, pooled_scores_spleen, ft_spleen], -1)
        )
    
        logits_bowel = torch.zeros(logits_spleen.size(), dtype=logits_spleen.dtype, device=logits_spleen.device)[:, :1]
        logits_extrav = torch.zeros(logits_spleen.size(), dtype=logits_spleen.dtype, device=logits_spleen.device)[:, :1]
        logits_kidney = torch.zeros(logits_spleen.size(), dtype=logits_spleen.dtype, device=logits_spleen.device)
        logits_liver = torch.zeros(logits_spleen.size(), dtype=logits_spleen.dtype, device=logits_spleen.device)

        logits = torch.cat(
            [logits_bowel, logits_extrav, logits_kidney, logits_liver, logits_spleen],
            -1,
        )

        return logits, torch.zeros((x.size(0)))
