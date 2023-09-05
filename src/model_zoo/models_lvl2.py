import torch
import torch.nn as nn


def define_model(name="rnn", ft_dim=2048, layer_dim=64, dense_dim=256, p=0.1, use_msd=False, num_classes=2, num_classes_aux=0, n_fts=0):
    if name == "rnn":
        return RNNModel(
            ft_dim=ft_dim,
            lstm_dim=layer_dim,
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
    elif name == "cnn":
        return CNNModel(
            ft_dim=ft_dim,
            cnn_dim=layer_dim,
            dense_dim=dense_dim,
            p=p,
            use_msd=use_msd,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_fts=n_fts,
            k=7,
            use_bn=True,
        )
    else:
        raise NotImplementedError


class RNNModel(nn.Module):
    def __init__(
        self,
        ft_dim=64,
        lstm_dim=64,
        dense_dim=64,
        p=0.1,
        use_msd=False,
        num_classes=8,
        num_classes_aux=0,
        n_fts=0,
    ):
        super().__init__()
        self.n_fts = n_fts
        self.use_msd = use_msd
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux

        self.mlp = nn.Sequential(
            nn.Linear(ft_dim, dense_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 4 + dense_dim * 2 + n_fts, num_classes),
        )

        if num_classes_aux:
            self.logits_aux = nn.Sequential(
                nn.Linear(lstm_dim * 2 + dense_dim + n_fts, num_classes_aux),
            )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, x, fts=None):
        features = self.mlp(x)
        features2, _ = self.lstm(features)

        features = torch.cat([features, features2], -1)

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
    
    
class CNNModel(nn.Module):
    def __init__(
        self,
        ft_dim=64,
        cnn_dim=64,
        dense_dim=64,
        kernel_size=3,
        p=0.1,
        use_msd=False,
        use_bn=True,
        k=3,
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
            nn.Linear(ft_dim, cnn_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
        )

        self.cnn = nn.Sequential(
            ConvBlock(cnn_dim, cnn_dim, kernel_size=k, use_bn=use_bn),
            nn.Dropout(p=p),
            ConvBlock(cnn_dim, cnn_dim, kernel_size=k, use_bn=use_bn),
            nn.Dropout(p=p),
        )

        self.logits = nn.Sequential(
            nn.Linear(cnn_dim * 2 * 4 + n_fts, num_classes),
        )

        self.high_dropout = nn.Dropout(p=0.5)

    def forward(self, x, fts=None):
        features = self.mlp(x)
    
        features = features.transpose(1, 2)
        features2 = self.cnn(features)

        features = torch.cat([features, features2], -1)
        pooled = features.view(features.size(0), -1)

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
