import copy
import json
import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn

from .models import MLP

logger = logging.getLogger(__name__)


class CostEstimator(nn.Module):
    meta_trained = False
    train_steps = 1000
    train_lr = 1e-3
    refine_steps = 50
    refine_lr = 1e-3

    def __init__(
        self,
        in_features: int | List[int],
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 32,
        enc_nlayers: int = 3,
        in_metafeat_dim: Optional[int] = None,
        out_metafeat_dim: int = 4,
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

        if in_metafeat_dim is not None:
            self.fc_meta = MLP(in_metafeat_dim, out_metafeat_dim)
            enc_dims += out_metafeat_dim
        else:
            self.fc_meta = None

        self.head = MLP(enc_dims, 1, enc_nlayers, enc_hidden_dim, act_fn=nn.GELU)

    @classmethod
    def from_pretrained(cls, root: str) -> "CostEstimator":
        config = json.load(open(os.path.join(root, "config.json"), "r"))
        model = cls(**config)
        model_path = os.path.join(root, "estimator.pth")
        model._load_meta_checkpoint(model_path)
        return model

    def forward(self, config, metafeat=None, **kwargs):
        # encode config
        x = []
        start = 0
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
        # x = nn.functional.relu(x)  # cost is always positive
        return x

    def _load_meta_checkpoint(self, path: str):
        self.meta_checkpoint = path
        self.meta_trained = True
        state = torch.load(path, map_location="cpu")
        msg = self.load_state_dict(state)
        logger.info(f"Loaded pretrained weights from {path} with msg: {msg}")

    def fit_pipeline(self, data: dict):
        state_dict = copy.deepcopy(self.state_dict())

        target = data.pop("cost")
        pred = self(**data)
        loss_pre = torch.nn.functional.l1_loss(pred, target)

        self.train()
        lr = self.refine_lr if self.meta_trained else self.train_lr
        optimizer = torch.optim.Adam(self.parameters(), lr)
        
        steps = self.refine_steps if self.meta_trained else self.train_steps
        for i in range(steps):
            optimizer.zero_grad()
            output = self(**data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        self.eval()
        pred = self(**data)
        loss_aft = torch.nn.functional.l1_loss(pred, target)

        if loss_aft > loss_pre:
            self.load_state_dict(state_dict)
            print("Refinement failed, reverting to previous state.")

        print(f"Accuracy before: {loss_pre:.4f}, after: {loss_aft:.4f}")