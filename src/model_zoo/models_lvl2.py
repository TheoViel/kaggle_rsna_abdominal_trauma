import torch
import torch.nn as nn


def define_model(name="rnn", ft_dim=2048, layer_dim=64, n_layers=1, dense_dim=256, p=0.1, use_msd=False, num_classes=2, num_classes_aux=0, n_fts=0):
    if name == "rnn":
        return RNNModel(
            ft_dim=ft_dim,
            lstm_dim=layer_dim,
            n_lstm=n_layers,
            dense_dim=dense_dim,
            p=p,
            use_msd=use_msd,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_fts=n_fts,
        )
    elif name == "rnn_att":
        return RNNAttModel(
            ft_dim=ft_dim,
            lstm_dim=layer_dim,
            n_lstm=n_layers,
            dense_dim=dense_dim,
            p=p,
            use_msd=use_msd,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_fts=n_fts,
        )
    elif name == "mlp":
        return MLPModel(
            ft_dim=ft_dim,
            dense_dim=dense_dim,
            p=p,
            use_msd=use_msd,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_fts=n_fts,
        )
    elif name == "cnn_att":
        return CNNAttModel(
            ft_dim=ft_dim,
            cnn_dim=layer_dim,
            dense_dim=dense_dim,
            p=p,
            use_msd=use_msd,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_fts=n_fts,
            use_bn=True,
            kernel_size=3,
        )
    else:
        raise NotImplementedError


class RNNModel(nn.Module):
    def __init__(
        self,
        ft_dim=64,
        lstm_dim=64,
        n_lstm=1,
        dense_dim=64,
        p=0.1,
        use_msd=False,
        num_classes=8,
        num_classes_aux=0,
        n_fts=0,
    ):
        super().__init__()
        self.n_fts = n_fts
        self.n_lstm = n_lstm
        self.use_msd = use_msd
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux

        self.mlp = nn.Sequential(
            nn.Linear(ft_dim, dense_dim),
            nn.Dropout(p=p),
            nn.Mish(),
        )

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)
        
        if self.n_lstm >= 2:
            self.lstm_2 = nn.LSTM(lstm_dim * 2, lstm_dim, batch_first=True, bidirectional=True)
        if self.n_lstm >= 3:
            self.lstm_3 = nn.LSTM(lstm_dim * 2, lstm_dim, batch_first=True, bidirectional=True)
        if self.n_lstm >= 4:
            raise NotImplementedError
    
        self.logits = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(lstm_dim * 4 + dense_dim * 2 + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, num_classes),
        )

        if num_classes_aux:
            self.logits_aux = nn.Sequential(
                nn.Dropout(p=p),
                nn.Linear(lstm_dim * 4 + dense_dim * 2 + n_fts, dense_dim),
                nn.Mish(),
                nn.Linear(dense_dim, num_classes_aux),
            )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, x, fts=None):
        features = self.mlp(x)
        features_lstm, _ = self.lstm(features)
        
        if self.n_lstm >= 2:
            features_lstm, _ = self.lstm_2(features_lstm)
        if self.n_lstm >= 3:
            features_lstm, _ = self.lstm_3(features_lstm)
        if self.n_lstm >= 4:
            raise NotImplementedError

        features = torch.cat([features, features_lstm], -1)

        mean = features.mean(1)
        max_, _ = features.max(1)
        pooled = torch.cat([mean, max_], -1)

        if fts is not None and self.n_fts:
            pooled = torch.cat([pooled, fts], -1)

        logits_aux = torch.zeros((x.size(0)))
        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(pooled)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
            if self.num_classes_aux:
                logits_aux = torch.mean(
                    torch.stack(
                        [self.logits_aux(self.high_dropout(features)) for _ in range(5)],
                        dim=0,
                    ),
                    dim=0,
                )
        else:
            logits = self.logits(pooled)
            if self.num_classes_aux:
                logits_aux = self.logits_aux(features)

        return logits, logits_aux


class RNNAttModel(nn.Module):
    def __init__(
        self,
        ft_dim=64,
        lstm_dim=64,
        n_lstm=1,
        dense_dim=64,
        p=0.1,
        use_msd=False,
        num_classes=8,
        num_classes_aux=0,
        n_fts=0,
    ):
        super().__init__()
        self.n_fts = n_fts
        self.n_lstm = n_lstm
        self.use_msd = use_msd
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux

        self.mlp = nn.Sequential(
            nn.Linear(ft_dim, dense_dim),
            nn.Dropout(p=p),
            nn.Mish(),
        )

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)
    
        self.logits_bowel = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 1),
        )
        self.logits_extrav = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 1),
        )
        self.logits_spleen = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )
        self.logits_liver = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )
        self.logits_kidney = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )

        if num_classes_aux:
            raise NotImplementedError
        
    def attention_pooling(self, x, w):
        return (x * w).sum(1) / w.sum(1), (x * w).amax(1)

    def forward(self, x, fts=None):
        features = self.mlp(x)
        features_lstm, _ = self.lstm(features)

        features = torch.cat([features, features_lstm], -1)
        
        kidney = x[:, :, 2: 4].amax(-1, keepdims=True)
        liver = x[:, :, :1]
        spleen = x[:, :, 1: 2]
        bowel = x[:, :, 4: 5]
        
        att_bowel, max_bowel = self.attention_pooling(features, bowel)
        att_kidney, max_kidney = self.attention_pooling(features, kidney)
        att_liver, max_liver = self.attention_pooling(features, liver)
        att_spleen, max_spleen = self.attention_pooling(features, spleen)

        mean = features.mean(1)
        max_ = features.amax(1)

        logits_bowel = self.logits_bowel(torch.cat([att_bowel, max_bowel], -1))
        logits_extrav = self.logits_extrav(torch.cat([mean, max_], -1))
        logits_kidney  = self.logits_kidney(torch.cat([att_kidney, max_kidney], -1))
        logits_liver = self.logits_liver(torch.cat([att_liver, max_liver], -1))
        logits_spleen = self.logits_spleen(torch.cat([att_spleen, max_spleen], -1))

        logits = torch.cat(
            [logits_bowel, logits_extrav, logits_kidney, logits_liver, logits_spleen],
            -1
        )

        return logits, torch.zeros((x.size(0)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding="same", use_bn=True):
        super().__init__()
        if padding == "same":
            padding = kernel_size // 2 * dilation
        
        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.ReLU(),
            )
                
    def forward(self, x):
        return self.conv(x)
    
    
class CNNAttModel(nn.Module):
    def __init__(
        self,
        ft_dim=64,
        cnn_dim=64,
        dense_dim=64,
        kernel_size=5,
        p=0.1,
        use_msd=False,
        use_bn=True,
        num_classes=2,
        num_classes_aux=0,
        n_fts=0
    ):
        super().__init__()
        self.ft_dim = ft_dim
        self.use_msd = use_msd
        self.n_fts = n_fts
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux

        self.mlp = nn.Sequential(
            nn.Linear(ft_dim, dense_dim),
#             nn.Dropout(p=p),
            nn.ReLU(),
        )

        self.cnn = nn.Sequential(
            ConvBlock(dense_dim, cnn_dim, kernel_size=kernel_size, use_bn=use_bn),
            nn.Dropout(p=p),
#             ConvBlock(cnn_dim, cnn_dim, kernel_size=kernel_size, use_bn=use_bn),
#             nn.Dropout(p=p),
#             ConvBlock(cnn_dim, cnn_dim, kernel_size=kernel_size, use_bn=use_bn),
#             nn.Dropout(p=p),
        )
    
        self.logits_bowel = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * (cnn_dim + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 1),
        )
        self.logits_extrav = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * (cnn_dim + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 1),
        )
        self.logits_spleen = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * (cnn_dim + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )
        self.logits_liver = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * (cnn_dim + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )
        self.logits_kidney = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * (cnn_dim + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )

        if num_classes_aux:
            raise NotImplementedError
        
    def attention_pooling(self, x, w):
        return (x * w).sum(1) / w.sum(1), (x * w).amax(1)

    def forward(self, x, fts=None):
        features = self.mlp(x)
        features2 = self.cnn(features.transpose(1, 2)).transpose(1, 2)
        features = torch.cat([features, features2], -1)

        kidney = x[:, :, 2: 4].amax(-1, keepdims=True)
        liver = x[:, :, :1]
        spleen = x[:, :, 1: 2]
        bowel = x[:, :, 4: 5]
        
        att_bowel, max_bowel = self.attention_pooling(features, bowel)
        att_kidney, max_kidney = self.attention_pooling(features, kidney)
        att_liver, max_liver = self.attention_pooling(features, liver)
        att_spleen, max_spleen = self.attention_pooling(features, spleen)

        mean = features.mean(1)
        max_ = features.amax(1)

        logits_bowel = self.logits_bowel(torch.cat([att_bowel, max_bowel], -1))
        logits_extrav = self.logits_extrav(torch.cat([mean, max_], -1))
        logits_kidney  = self.logits_kidney(torch.cat([att_kidney, max_kidney], -1))
        logits_liver = self.logits_liver(torch.cat([att_liver, max_liver], -1))
        logits_spleen = self.logits_spleen(torch.cat([att_spleen, max_spleen], -1))

        logits = torch.cat(
            [logits_bowel, logits_extrav, logits_kidney, logits_liver, logits_spleen],
            -1
        )
        return logits, torch.zeros((x.size(0)))
    
        
class MLPModel(nn.Module):
    def __init__(
        self,
        ft_dim=2,
        dense_dim=64,
        p=0.1,
        use_msd=False,
        num_classes=2,
        num_classes_aux=0,
        n_fts=0
    ):
        super().__init__()
        self.ft_dim = ft_dim
        self.use_msd = use_msd
        self.n_fts = n_fts
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux

        self.mlp = nn.Sequential(
            nn.Linear(ft_dim, dense_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
        )
        
        self.mlp_2 = nn.Sequential(
            nn.Linear(dense_dim * 4, dense_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(dense_dim * 2, dense_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
        )
        
        self.logits = nn.Sequential(
            nn.Linear(dense_dim + n_fts, num_classes),
        )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, x, fts=None):
        features = self.mlp(x)

        pooled = features.view(features.size(0), -1)
        
        pooled = self.mlp_2(pooled)

        if fts is not None and self.n_fts:
            pooled = torch.cat([pooled, fts], -1)

        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(pooled)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            logits = self.logits(pooled)

        return logits
