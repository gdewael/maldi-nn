import torch.nn as nn
import torch
from bio_attention.attention import TransformerEncoder
from bio_attention.embed import ContinuousEmbedding


class MLP(nn.Module):
    def __init__(
        self,
        n_inputs=6000,
        n_outputs=512,
        layer_dims=[512, 512, 512],
        layer_or_batchnorm="layer",
        dropout=0.2,
    ):
        super().__init__()

        c = n_inputs
        layers = []
        for i in layer_dims:
            layers.append(nn.Linear(c, i))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(
                nn.LayerNorm(i) if layer_or_batchnorm == "layer" else nn.BatchNorm1d(i)
            )
            c = i

        layers.append(nn.Linear(c, n_outputs))

        self.net = nn.Sequential(*layers)

        self.hsize = n_outputs

    def forward(self, spectrum):
        return self.net(spectrum["intensity"])


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)


class Transformer(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        n_heads=8,
        dropout=0.2,
        cls=True,
        reduce="none",
        output_head_dim=64,
    ):
        super().__init__()

        self.embed = ContinuousEmbedding(
            dim,
            depth=1,
            norm=False,
            cls=cls,
            init_cls_as_bias=True,
            init_mask_as_bias=True,
        )

        self.transformer = TransformerEncoder(
            depth=depth,
            dim=dim,
            nh=n_heads,
            attentiontype="vanilla",
            attention_args={"dropout": dropout},
            plugintype="sinusoidal",
            plugin_args={"dim": dim, "divide": 10},
            only_apply_plugin_at_first=True,
            dropout=dropout,
            glu_ff=True,
            activation="gelu",
        )

        self.reduce = reduce

        self.output_head = nn.Linear(dim, output_head_dim)

    def forward(self, spectrum):
        z = self.embed(spectrum["intensity"])
        z = self.transformer(z, pos=spectrum["mz"])

        if self.reduce == "mean":
            return self.output_head(z.sum(1))
        elif self.reduce == "max":
            return self.output_head(z.max(1).values)
        elif self.reduce == "cls":
            return self.output_head(z[:, 0, :])
        elif self.reduce == "none":
            return z
