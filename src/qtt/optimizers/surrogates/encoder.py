from typing import List, Optional

import torch
import torch.nn as nn

from .models import MLP, ConvNet


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        in_features: int | List[int],
        out_features: int = 32,
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 32,
        enc_nlayers: int = 3,
        in_curve_dim: int = 1,
        out_curve_dim: int = 16,
        in_metafeat_dim: Optional[int] = None,
        out_metafeat_dim: int = 16,
    ):
        super().__init__()
        if isinstance(in_features, int):
            in_features = [in_features]
        self.in_features = in_features
        enc_dims = 0  # feature dimension after encoding

        # build config encoder
        encoder = nn.ModuleList()
        for dim in in_features:
            encoder.append(MLP(dim, enc_out_dim, enc_nlayers, enc_hidden_dim))
        self.config_encoder = encoder
        enc_dims = len(in_features) * enc_out_dim

        # add 1 dim for budget
        enc_dims += 1

        # build curve encoder
        self.curve_embedder = ConvNet(in_curve_dim, out_curve_dim)
        enc_dims += out_curve_dim

        if in_metafeat_dim is not None:
            enc_dims += out_metafeat_dim
            self.fc_meta = nn.Linear(in_metafeat_dim, out_metafeat_dim)
        else:
            self.fc_meta = None

        self.head = MLP(enc_dims, out_features, 3, enc_hidden_dim, act_fn=nn.GELU)

    def forward(self, config, budget, curve, metafeat=None, **kwargs):
        # encode config
        start = 0
        x = []
        for i, dim in enumerate(self.in_features):
            end = start + dim
            output = self.config_encoder[i](config[:, start:end])  # type: ignore
            x.append(output)
            start = end
        x = torch.cat(x, dim=1)

        # concatenate budget
        if budget.dim() == 1:
            budget = torch.unsqueeze(budget, dim=1)
        x = torch.cat([x, budget], dim=1)

        # encode curve
        if curve.dim() == 2:
            curve = torch.unsqueeze(curve, dim=1)
        curve = self.curve_embedder(curve)
        x = torch.cat([x, curve], dim=1)

        # encode meta-features
        if self.fc_meta is not None:
            out = self.fc_meta(metafeat)
            x = torch.cat([x, out], dim=1)

        x = self.head(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
