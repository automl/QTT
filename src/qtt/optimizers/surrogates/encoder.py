import torch
import torch.nn as nn

from .models import MLP, CNN


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int | list[int],
        out_dim: int = 32,
        enc_hidden_dim: int = 128,
        enc_out_dim: int = 32,
        enc_nlayers: int = 3,
        in_curve_dim: int = 50,
        out_curve_dim: int = 16,
        curve_channels: int = 1,
        in_metafeat_dim: int | None = None,
        out_metafeat_dim: int = 16,
    ):
        super().__init__()
        if isinstance(in_dim, int):
            in_dim = [in_dim]
        self.in_dim = in_dim
        enc_dims = 0  # feature dimension after encoding

        # build config encoder
        encoder = nn.ModuleList()
        for dim in in_dim:
            encoder.append(MLP(dim, enc_out_dim, enc_nlayers, enc_hidden_dim))
        self.config_encoder = encoder
        enc_dims = len(in_dim) * enc_out_dim

        # add 1 dim for fidelity
        enc_dims += 1

        # build curve encoder
        self.curve_encoder = CNN(in_curve_dim, curve_channels, out_curve_dim)
        enc_dims += out_curve_dim

        if in_metafeat_dim is not None:
            enc_dims += out_metafeat_dim
            self.fc_meta = nn.Linear(in_metafeat_dim, out_metafeat_dim)
        else:
            self.fc_meta = None

        self.head = MLP(enc_dims, out_dim, 3, enc_hidden_dim, act_fn=nn.GELU)

    def forward(self, config, fidelity, curve, metafeat=None, **kwargs):
        # encode config
        start = 0
        x = []
        for i, dim in enumerate(self.in_dim):
            end = start + dim
            output = self.config_encoder[i](config[:, start:end])  # type: ignore
            x.append(output)
            start = end
        x = torch.cat(x, dim=1)

        # concatenate fidelity
        if fidelity.dim() == 1:  # BS 
            fidelity = torch.unsqueeze(fidelity, dim=1) # BS x 1
        x = torch.cat([x, fidelity], dim=1)

        # encode curve
        if curve.dim() == 2:  # BS x curve_dim
            curve = torch.unsqueeze(curve, dim=1)  # BS x 1 x curve_dim
        curve = self.curve_encoder(curve)
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
