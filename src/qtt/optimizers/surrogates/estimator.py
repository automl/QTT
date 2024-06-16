import json
import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn

from .models import MLP

logger = logging.getLogger("dyhpo")


class CostEstimator(nn.Module):
    def __init__(
        self,
        in_features: int | List[int],
        out_features: int = 1,
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 32,
        enc_nlayers: int = 3,
        in_meta_features: Optional[int] = None,
        out_meta_features: int = 16,
    ):
        super().__init__()
        if isinstance(in_features, int):
            in_features = [in_features]
        self.in_features = in_features
        enc_dims = 0  # feature dimension after encoding

        # build config encoder
        if isinstance(in_features, list):
            encoder = nn.ModuleList()
            for dim in in_features:
                encoder.append(MLP(dim, enc_out_dim, enc_nlayers, enc_hidden_dim))
            self.config_encoder = encoder
            enc_dims = len(in_features) * enc_out_dim
        else:
            self.config_encoder = MLP(
                in_features, enc_out_dim, enc_nlayers, enc_hidden_dim
            )
            enc_dims = enc_out_dim

        if in_meta_features is not None:
            enc_dims += out_meta_features
            self.fc_meta = nn.Linear(in_meta_features, out_meta_features)
        else:
            self.fc_meta = None

        self.head = MLP(enc_dims, out_features, 1, enc_hidden_dim, act_fn=nn.GELU)

    @classmethod
    def from_pretrained(cls, root: str) -> "CostEstimator":
        config = json.load(open(os.path.join(root, "config.json"), "r"))
        model = cls(**config)
        model_path = os.path.join(root, "estimator.pth")
        model.load_meta_checkpoint(model_path)
        return model

    def forward(self, config, metafeat=None, **kwargs):
        # encode config
        start = 0
        x = []
        for i, dim in enumerate(self.in_features):
            end = start + dim
            output = self.config_encoder[i](config[:, start:end])  # type: ignore
            x.append(output)
            start = end
        x = torch.cat(x, dim=1)

        if self.fc_meta is not None:
            out = self.fc_meta(metafeat)
            x = torch.cat([x, out], dim=1)

        x = self.head(x)
        x = nn.functional.relu(x)  # cost is always positive
        return x

    def load_meta_checkpoint(self, path: str):
        self.meta_checkpoint = path
        self.meta_trained = True
        state = torch.load(path, map_location="cpu")
        msg = self.load_state_dict(state)
        logger.info(f"Loaded pretrained weights from {path} with msg: {msg}")
