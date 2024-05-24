from typing import List, Optional

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, in_channels: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 8, 3, 1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(8, 8, 3, 1, padding="same"),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(200, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        out_features: int = 32,
        in_curve_dim: int = 1,
        out_curve_dim: int = 16,
        in_meta_features: int = 4,
        out_meta_features: int = 16,
        enc_num_layers: int = 2,
        enc_slice_ranges: Optional[List[int,]] = None,
    ):
        super().__init__()
        if enc_slice_ranges is not None:
            assert enc_slice_ranges[-1] < in_features
            _slices = [0] + enc_slice_ranges + [in_features]
            _ranges = [(_slices[i], _slices[i + 1]) for i in range(len(_slices) - 1)]
            self.enc_slice_ranges = _ranges
            self.encoder = self._build_encoder(hidden_features, enc_num_layers, _ranges)
            out_enc_features = len(self.encoder) * hidden_features
        else:
            out_enc_features = in_features
            self.encoder = None
        out_enc_features += out_curve_dim + out_meta_features + 1

        self.fc1 = nn.Linear(out_enc_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.curve_embedder = ConvNet(in_curve_dim, out_curve_dim)
        self.fc_meta = nn.Linear(in_meta_features, out_meta_features)

        self.act = nn.LeakyReLU()

    def _build_encoder(self, hidden_feat, num_layers, ranges):
        encoder = nn.ModuleList()
        for a, b in ranges:
            encoder.append(MLP(b - a, [hidden_feat] * num_layers, hidden_feat))
        return encoder

    def forward(self, config, budget, curve, metafeat=None):
        budget = torch.unsqueeze(budget, dim=1)
        if self.encoder is not None:
            x = []
            for (a, b), encoder in zip(self.enc_slice_ranges, self.encoder):
                x.append(encoder(config[:, a:b]))
            x.append(budget)
            x = torch.cat(x, dim=1)
        else:
            x = torch.cat([config, budget], dim=1)

        if metafeat is not None:
            out = self.fc_meta(metafeat)
            x = torch.cat([x, out], dim=1)

        curve = torch.unsqueeze(curve, dim=1)
        curve = self.curve_embedder(curve)
        x = torch.cat([x, curve], dim=1)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class CostPredictor(FeatureExtractor):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        out_features = kwargs.get("out_features", 32)
        self.act = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(out_features, 1)

    @classmethod
    def init_from_config(cls, config: dict) -> "CostPredictor":
        return cls(**config)

    def forward(self, config, budget, curve, metafeat=None):
        x = super().forward(config, budget, curve, metafeat)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | list[int], out_features: int):
        super().__init__()
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features]
        layers = [nn.Linear(in_features, hidden_features[0]), nn.ReLU()]
        for i in range(len(hidden_features) - 1):
            in_ftr, out_ftr = hidden_features[i], hidden_features[i + 1]
            layers.append(nn.Linear(in_ftr, out_ftr))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_features[-1], out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
