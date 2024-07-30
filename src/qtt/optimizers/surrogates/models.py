from typing import Type

import torch.nn as nn


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


if __name__ == "__main__":
    import torch

    # test CNN
    cnn = CNN(in_channels=1, in_dim=50, out_dim=16)
    x = torch.randn(64, 1, 50)
    y = cnn(x)
    print(y.shape)

    # test MLP
    mlp = MLP(in_dim=50, out_dim=16)
    x = torch.randn(10, 50)
    y = mlp(x)
    print(y.shape)
