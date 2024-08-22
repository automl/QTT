from typing import Type

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int | list[int],
        in_curve_dim: int,
        out_dim: int = 16,
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 8,
        enc_nlayers: int = 3,
        out_curve_dim: int = 8,
    ):
        super().__init__()
        if isinstance(in_dim, int):
            in_dim = [in_dim]
        self.in_dim = in_dim
        enc_dims = 0  # feature dimension after encoding

        # build pipeline encoder
        encoder = nn.ModuleList()
        for dim in self.in_dim:
            encoder.append(MLP(dim, enc_out_dim, enc_nlayers, enc_hidden_dim))
        self.config_encoder = encoder
        enc_dims = len(self.config_encoder) * enc_out_dim

        # build curve encoder
        self.curve_encoder = MLP(in_curve_dim, out_curve_dim, enc_nlayers, enc_hidden_dim)
        enc_dims += out_curve_dim

        self.head = MLP(enc_dims, out_dim, 3, enc_hidden_dim, act_fn=nn.GELU)

    def forward(self, pipeline, curve):
        # encode config
        start = 0
        x = []
        for i, dim in enumerate(self.in_dim):
            end = start + dim
            output = self.config_encoder[i](pipeline[:, start:end])  # type: ignore
            x.append(output)
            start = end
        x = torch.cat(x, dim=1)

        # encode curve
        out = self.curve_encoder(curve)
        x = torch.cat([x, out], dim=1)

        x = self.head(x)
        # x = torch.softmax(x, dim=-1)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class CNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        in_channels: int,
        out_dim: int,
        act_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=8, kernel_size=3, padding="same"),
            act_fn(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(8, out_channels=16, kernel_size=3, padding="same"),
            act_fn(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, out_channels=32, kernel_size=3, padding="same"),
            act_fn(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * (in_dim // 8), 32 * (in_dim // 8) // 2),
            act_fn(),
            nn.Linear(32 * (in_dim // 8) // 2, out_dim),
        )

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        nlayers: int = 2,
        hidden_dim: int = 128,
        bottleneck_dim: int = 8,
        act_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), act_fn()]
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(bottleneck_dim, out_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.head(x)
        return x
