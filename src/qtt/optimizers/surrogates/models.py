from typing import Type

import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 8, 3, 1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(8, 8, 3, 1, padding="same"),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(200, out_dim),
            nn.ReLU(),
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
        self.out = nn.Linear(bottleneck_dim, out_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.out(x)
        return x
