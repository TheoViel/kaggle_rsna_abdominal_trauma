import torch
import torch.nn as nn

from transformers import AutoConfig
from model_zoo.seq import DebertaV2Output
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder



def define_model(
    name="rnn_att",
    ft_dim=2048,
    layer_dim=64,
    n_layers=1,
    dense_dim=256,
    p=0.1,
    use_msd=False,
    num_classes=2,
    num_classes_aux=0,
    n_fts=0,
    use_other_series=False,
):
    if name == "rnn_att":
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
            use_other_series=use_other_series,
        )
    elif name == "transfo":
        return TransfoModel(
            ft_dim=ft_dim,
            transfo_dim=layer_dim,
            n_lstm=n_layers,
            dense_dim=dense_dim,
            p=p,
            use_msd=use_msd,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_fts=n_fts,
        )
    else:
        raise NotImplementedError


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
        use_other_series=False,
    ):
        super().__init__()
        self.n_fts = n_fts
        self.n_lstm = n_lstm
        self.use_msd = use_msd
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.use_other_series = use_other_series

        self.mlp = nn.Sequential(
            nn.Linear(ft_dim, dense_dim),
#             nn.Dropout(p=p),
            nn.Mish(),
        )
        
        if n_fts > 0:
            self.mlp_fts = nn.Sequential(
                nn.Linear(n_fts, dense_dim),
                nn.Dropout(p=p),
                nn.Mish(),
            )
            n_fts = n_fts // 3 + dense_dim
            
#         if use_other_series:
#             self.mlp_other= nn.Sequential(
#                 nn.Linear(ft_dim, dense_dim),
#     #             nn.Dropout(p=p),
#                 nn.Mish(),
#             )
#             self.lstm_other = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

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
        self.logits_spleen = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 8 + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )
        self.logits_liver = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 8 + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )
        self.logits_kidney = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(2 * (lstm_dim * 2 + dense_dim) + 8 + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )

        if num_classes_aux:
            raise NotImplementedError
        
    def attention_pooling(self, x, w):
        return (x * w).sum(1) / (w.sum(1) + 1e-6), (x * w).amax(1)

    def forward(self, x, ft=None, x_other=None):
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
                        
        scores = x[:, :, 5:]
#         scores = scores[:, :, :scores.size(2) // 22 * 22]
        scores = scores.view(x.size(0), x.size(1), -1, 11 * 2)
        pooled_scores = scores.mean(2).view(x.size(0), x.size(1), 2, 11)
        pooled_scores = torch.cat([pooled_scores.amax(1), pooled_scores.mean(1)], 1)
        
#         if (x.size(2) - 5) % 22:
#             score_be = x[:, :, - ((x.size(2) - 5) % 22):]
# #             print(score_be.size())
#             score_be = score_be.view(x.size(0), x.size(1), -1, 2 * 2)
#             pooled_scores_be = score_be.mean(2).view(x.size(0), x.size(1), 2, 2)
#             pooled_scores_be = torch.cat([pooled_scores_be.amax(1), pooled_scores_be.mean(1)], 1)
        
# #             print(pooled_scores_be.size())
# #             print(pooled_scores.size())
#             pooled_scores[:, :, 0] = torch.amax(
#                 torch.stack([pooled_scores[:, :, 0] , pooled_scores_be[:, :, 0] ], 0), 0
#             )
#             pooled_scores[:, :, 1] = torch.mean(
#                 torch.stack([pooled_scores[:, :, 1] , pooled_scores_be[:, :, 1] ], 0), 0
#             )

        pooled_scores_bowel = pooled_scores[:, :, :1].flatten(1, 2)
        pooled_scores_extrav = pooled_scores[:, :, 1: 2].flatten(1, 2)
        pooled_scores_kidney = pooled_scores[:, :, 3: 5].flatten(1, 2)
        pooled_scores_liver = pooled_scores[:, :, 6: 8].flatten(1, 2)
        pooled_scores_spleen = pooled_scores[:, :, 9: 11].flatten(1, 2)

        if (x_other is not None) and self.use_other_series:
#             features_other = self.mlp_other(x_other)
#             features_lstm_other, _ = self.lstm_other(features_other)
#             features_other = torch.cat([features_other, features_lstm_other], -1)

#             mean_other = features_other.mean(1)
#             max_other = features_other.amax(1)
            
#             mean = torch.mean(torch.stack([mean, mean_other], 0), 0)
#             max_ = torch.amax(torch.stack([max_, max_other], 0), 0)
            
#             if x_other.size(-1) == x.size(-1):  # Merge everything
#                 kidney = x_other[:, :, 2: 4].amax(-1, keepdims=True)
#                 liver = x_other[:, :, :1]
#                 spleen = x_other[:, :, 1: 2]
#                 bowel = x_other[:, :, 4: 5]

#                 att_bowel_o, max_bowel_o = self.attention_pooling(features_other, bowel)
#                 att_kidney_o, max_kidney_o = self.attention_pooling(features_other, kidney)
#                 att_liver_o, max_liver_o = self.attention_pooling(features_other, liver)
#                 att_spleen_o, max_spleen_o = self.attention_pooling(features_other, spleen)

#                 att_bowel = torch.mean(torch.stack([att_bowel, att_bowel_o], 0), 0)
#                 max_bowel  = torch.amax(torch.stack([max_bowel, max_bowel_o], 0), 0)
#                 att_kidney = torch.mean(torch.stack([att_kidney, att_kidney_o], 0), 0)
#                 max_kidney  = torch.amax(torch.stack([max_kidney, max_kidney_o], 0), 0)
#                 att_liver = torch.mean(torch.stack([att_liver, att_liver_o], 0), 0)
#                 max_liver  = torch.amax(torch.stack([max_liver, max_liver_o], 0), 0)
#                 att_spleen = torch.mean(torch.stack([att_spleen, att_spleen_o], 0), 0)
#                 max_spleen  = torch.amax(torch.stack([max_spleen, max_spleen_o], 0), 0)
                
            scores_o = x_other[:, :, 5:].view(x.size(0), x.size(1), -1, 11 * 2)
            pooled_scores_o = scores_o.mean(2).view(x.size(0), x.size(1), 2, 11)
            pooled_scores_o = torch.cat([pooled_scores_o.amax(1), pooled_scores_o.mean(1)], 1)

            pooled_scores_bowel_o = pooled_scores_o[:, :, :1].flatten(1, 2)
            pooled_scores_extrav_o = pooled_scores_o[:, :, 1: 2].flatten(1, 2)
            pooled_scores_kidney_o = pooled_scores_o[:, :, 3: 5].flatten(1, 2)
            pooled_scores_liver_o = pooled_scores_o[:, :, 6: 8].flatten(1, 2)
            pooled_scores_spleen_o = pooled_scores_o[:, :, 9: 11].flatten(1, 2)
            
            pooled_scores_bowel = torch.amax(torch.stack([pooled_scores_bowel, pooled_scores_bowel_o], 0), 0)
            pooled_scores_extrav = torch.amax(torch.stack([pooled_scores_extrav, pooled_scores_extrav_o], 0), 0)
            pooled_scores_kidney = torch.amax(torch.stack([pooled_scores_kidney, pooled_scores_kidney_o], 0), 0)
            pooled_scores_liver = torch.amax(torch.stack([pooled_scores_liver, pooled_scores_liver_o], 0), 0)
            pooled_scores_spleen = torch.amax(torch.stack([pooled_scores_spleen, pooled_scores_spleen_o], 0), 0)
        
        if ft is None or self.n_fts == 0:
            ft_kidney = torch.empty((x.size(0), 0)).to(x.device)
            ft_liver = torch.empty((x.size(0), 0)).to(x.device)
            ft_spleen = torch.empty((x.size(0), 0)).to(x.device)
        else:
            fts = self.mlp_fts(ft.flatten(1, 2))
            ft_kidney = torch.cat([fts, ft[:, 0]], -1)
            ft_liver = torch.cat([fts, ft[:, 1]], -1)
            ft_spleen = torch.cat([fts, ft[:, 2]], -1)

        logits_bowel = self.logits_bowel(torch.cat([att_bowel, max_bowel, pooled_scores_bowel], -1))
        logits_extrav = self.logits_extrav(torch.cat([mean, max_, pooled_scores_extrav], -1))
        logits_kidney  = self.logits_kidney(torch.cat([att_kidney, max_kidney, pooled_scores_kidney, ft_kidney], -1))
        logits_liver = self.logits_liver(torch.cat([att_liver, max_liver, pooled_scores_liver, ft_liver], -1))
        logits_spleen = self.logits_spleen(torch.cat([att_spleen, max_spleen, pooled_scores_spleen, ft_spleen], -1))

        logits = torch.cat(
            [logits_bowel, logits_extrav, logits_kidney, logits_liver, logits_spleen],
            -1
        )

        return logits, torch.zeros((x.size(0)))


class TransfoModel(nn.Module):
    def __init__(
        self,
        ft_dim=64,
        transfo_dim=64,
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
#             nn.Dropout(p=p),
            nn.Mish(),
        )

        self.cnn = nn.Sequential(
            ConvBlock(dense_dim, transfo_dim, kernel_size=3, use_bn=False),
            nn.Dropout(p=p),
        )

        name = "microsoft/deberta-v3-base"
        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim + dense_dim
        config.intermediate_size = transfo_dim + dense_dim
        config.output_size = transfo_dim
        config.num_hidden_layers = 1
        config.num_attention_heads = 8
        config.attention_probs_dropout_prob = p
        config.hidden_dropout_prob = p
        config.hidden_act = nn.Mish()
        config.max_relative_positions = 600
        config.position_buckets = 600
        config.skip_output = True

        self.transfo_bowel = DebertaV2Encoder(config)
        self.transfo_bowel.layer[0].output = DebertaV2Output(config)
        
        self.transfo_spleen = DebertaV2Encoder(config)
        self.transfo_spleen.layer[0].output = DebertaV2Output(config)

        self.transfo_liver = DebertaV2Encoder(config)
        self.transfo_liver.layer[0].output = DebertaV2Output(config)

        self.transfo_kidney = DebertaV2Encoder(config)
        self.transfo_kidney.layer[0].output = DebertaV2Output(config)
    
        self.logits_bowel = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * transfo_dim + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 1),
        )
        self.logits_extrav = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * (transfo_dim + dense_dim) + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 1),
        )
        self.logits_spleen = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * transfo_dim + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )
        self.logits_liver = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * transfo_dim + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )
        self.logits_kidney = nn.Sequential(
#             nn.Dropout(p=p),
            nn.Linear(2 * transfo_dim + n_fts, dense_dim),
            nn.Mish(),
            nn.Linear(dense_dim, 3),
        )

        if num_classes_aux:
            raise NotImplementedError
        
    def attention_pooling(self, x, w):
        return (x * w).sum(1) / (w.sum(1) + 1e-6), (x * w).amax(1)

    def forward(self, x, fts=None):
        features = self.mlp(x)
        
        features2 = self.cnn(features.transpose(1, 2)).transpose(1, 2)
        features = torch.cat([features, features2], -1)
        

        kidney = x[:, :, 2: 4].amax(-1)
#         kidney = x[:, :, 2]
        liver = x[:, :, 0]
        spleen = x[:, :, 1]
        bowel = x[:, :, 4]  # 3
 
        features_bowel = self.transfo_bowel(features, bowel).last_hidden_state
        features_kidney = self.transfo_kidney(features, kidney).last_hidden_state
        features_liver = self.transfo_liver(features, liver).last_hidden_state
        features_spleen = self.transfo_spleen(features, spleen).last_hidden_state

        att_bowel, max_bowel = self.attention_pooling(features_bowel, bowel.unsqueeze(-1))
        att_kidney, max_kidney = self.attention_pooling(features_kidney, kidney.unsqueeze(-1))
        att_liver, max_liver = self.attention_pooling(features_liver, liver.unsqueeze(-1))
        att_spleen, max_spleen = self.attention_pooling(features_spleen, spleen.unsqueeze(-1))

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
