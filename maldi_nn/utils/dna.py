import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import numpy as np
from maldi_nn.models import MaldiLightningModule
import h5torch
from maldi_nn.utils.data import MaldiDataModule

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class Permute(nn.Module): 
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return x.permute(*self.args)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-1], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[..., -trim:trim]

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1)



class EnformerStem(nn.Module):
    def __init__(
        self,
        input_hsize = 4, 
        C = 512, 
        ):
        """
        This model structure follows the Enformer model stack, up until the transformer part:
        https://www.nature.com/articles/s41592-021-01252-x/figures/5
        Code inspired and taken from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py.
        """
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(input_hsize, C // 2, 15, padding = 15 // 2),
            Residual(ConvBlock(C // 2, kernel_size = 1)),
            AttentionPool(C // 2),
        )

    def forward(self, x):
        return self.stem(x)


class EnformerConvTowerEncoder(nn.Module):
    def __init__(
        self,
        C = 512, # the last conv will have this many channels
        n_blocks = 6, # how many blocks there will be in the tower
        kernel_size = 5, # kernel size of the conv blocks
        ):
        """
        This model structure follows the Enformer model stack, up until the transformer part:
        https://www.nature.com/articles/s41592-021-01252-x/figures/5
        Code inspired and taken from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py.
        """
        super().__init__()


        factor = (C / (C // 2))**(1/n_blocks)
        hidden_sizes = [C // 2]
        for _ in range(n_blocks-1):
            hidden_sizes.append(int(hidden_sizes[-1] * factor))
        hidden_sizes.append(C)


        layers = []
        for i in range(1, len(hidden_sizes)):
            layers.append(ConvBlock(hidden_sizes[i-1], dim_out = hidden_sizes[i], kernel_size = kernel_size))
            layers.append(Residual(ConvBlock(hidden_sizes[i], kernel_size = 1)))
            if ((i+1) != len(hidden_sizes)):
                layers.append(AttentionPool(hidden_sizes[i]))
        

        self.conv_tower = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_tower(x)


class SmallMLPEncoder(nn.Module):
    def __init__(self, in_size, out_size, dropout = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, out_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_size * 4, out_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_size * 2, out_size),
        )

    def forward(self, x):
        return self.net(x)



class SmallMLPDecoder(nn.Module):
    def __init__(self, in_size, out_size, dropout = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, in_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_size * 2, in_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_size * 4, out_size),
        )

    def forward(self, x):
        return self.net(x)

class EnformerConvTowerDecoder(nn.Module):
    def __init__(
        self,
        C = 512, # the last conv will have this many channels
        n_blocks = 6, # how many blocks there will be in the tower
        kernel_size = 5, # kernel size of the conv blocks
        ):
        """
        This model structure follows the Enformer model stack, up until the transformer part:
        https://www.nature.com/articles/s41592-021-01252-x/figures/5
        Code inspired and taken from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py.
        """
        super().__init__()


        factor = (C / (C // 2))**(1/n_blocks)
        hidden_sizes = [C // 2]
        for _ in range(n_blocks-1):
            hidden_sizes.append(int(hidden_sizes[-1] * factor))
        hidden_sizes.append(C)
        hidden_sizes.reverse()


        layers = []
        for i in range(1, len(hidden_sizes)):
            layers.append(ConvBlock(hidden_sizes[i-1], dim_out = hidden_sizes[i], kernel_size = kernel_size))
            layers.append(Residual(ConvBlock(hidden_sizes[i], kernel_size = 1)))
            if ((i+1) != len(hidden_sizes)):
                layers.append(nn.ConvTranspose1d(hidden_sizes[i], hidden_sizes[i], kernel_size=2, stride=2))

        self.conv_tower = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_tower(x)
    

class EnformerHead(nn.Module):
    def __init__(
        self,
        output_hsize = 32, 
        C = 512, 
        ):
        """
        This model structure follows the Enformer model stack, up until the transformer part:
        https://www.nature.com/articles/s41592-021-01252-x/figures/5
        Code inspired and taken from https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py.
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.ConvTranspose1d(C // 2, C // 2, kernel_size=2, stride=2),
            Residual(ConvBlock(C // 2, kernel_size = 1)),
            nn.Conv1d(C // 2, output_hsize, 15, padding = 15 // 2),
        )

    def forward(self, x):
        return self.head(x)
    


class rRNAEncoder(nn.Module):
    def __init__(
        self,
        input_hsize = 32, 
        C = 256, # the last conv will have this many channels
        n_blocks = 12, # how many blocks there will be in the tower
        kernel_size = 5, # kernel size of the conv blocks
        output_hsize = 64,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Embedding(17, input_hsize),
            Permute(0,2,1), 
            EnformerStem(input_hsize = input_hsize, C = C),
            EnformerConvTowerEncoder(C = C, n_blocks = n_blocks, kernel_size = kernel_size),
            Rearrange('b c l -> b (c l)'),
            SmallMLPEncoder(int(np.ceil(50_000 / 2**n_blocks) * C), output_hsize),
        )
    def forward(self, x):
        return self.encoder(x)

class rRNADecoder(nn.Module):
    def __init__(
        self,
        input_hsize = 64,
        C = 256, # the last conv will have this many channels
        n_blocks = 12, # how many blocks there will be in the tower
        kernel_size = 5, # kernel size of the conv blocks
        output_hsize = 32,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            SmallMLPDecoder(input_hsize, int(np.ceil(50_000 / 2**n_blocks) * C)),
            Rearrange('b (c l) -> b c l', c=C),
            EnformerConvTowerDecoder(C=C, n_blocks=n_blocks, kernel_size=kernel_size),
            EnformerHead(output_hsize=output_hsize, C=C),
            TargetLengthCrop(50_000),
            Permute(0,2,1),
            nn.Linear(output_hsize, 17),
            Permute(0,2,1),
        )
    def forward(self, x):
        return self.decoder(x)


class rRNADataModule(MaldiDataModule):
    def __init__(
        self,
        path,
        batch_size=512,
        n_workers=4,
        in_memory=True,
    ):
        super().__init__(batch_size=batch_size, n_workers=n_workers)
        self.path = path
        self.in_memory = in_memory

    def setup(self, stage):
        f = h5torch.File(self.path)

        if self.in_memory:
            f = f.to_dict()

        self.train = h5torch.Dataset(
            f, subset=("0/split", "train"), sample_processor=None
        )

        self.val = h5torch.Dataset(
            f, subset=("0/split", "val"), sample_processor=None
        )

        self.test = h5torch.Dataset(
            f, subset=("0/split", "test"), sample_processor=None
        )


class rRNAVAE(MaldiLightningModule):
    def __init__(
        self,
        input_hsize = 32,
        C = 256, # the last conv will have this many channels
        n_blocks = 12, # how many blocks there will be in the tower
        kernel_size = 5, # kernel size of the conv blocks
        bottleneck_size = 64,
        beta = 0.001,
        lr=1e-4,
        weight_decay=0,
        lr_decay_factor=1.00,
        warmup_steps=250,
    ):
        super().__init__(
            lr=lr,
            weight_decay=weight_decay,
            lr_decay_factor=lr_decay_factor,
            warmup_steps=warmup_steps,
        )
        self.save_hyperparameters()

        self.encoder = rRNAEncoder(
            input_hsize = input_hsize, 
            C = C, 
            n_blocks = n_blocks,
            kernel_size = kernel_size,
            output_hsize = bottleneck_size,
        )

        self.decoder = rRNADecoder(
            input_hsize = bottleneck_size,
            C = C, 
            n_blocks = n_blocks,
            kernel_size = kernel_size,
            output_hsize = input_hsize,
        )

        self.to_mu = nn.Linear(bottleneck_size, bottleneck_size)
        self.to_var = nn.Linear(bottleneck_size, bottleneck_size)
        self.beta = beta
        
    def forward(self, x):
        z = self.encoder(x)
        z_means, z_logvars = self.to_mu(z), self.to_var(z)

        z = self.reparameterization(z_means, z_logvars)
        x_reconstruct = self.decoder(z)

        return x_reconstruct, z_means, z_logvars

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def loss(self, x_reconstruct, means, logvars, x_true):
        reconstruction_loss = F.cross_entropy(x_reconstruct, x_true)

        KL_div_loss = (-0.5 * torch.sum(1 + logvars - means**2 - logvars.exp())) / means.size(1)
        return reconstruction_loss + self.beta * KL_div_loss
    
    def training_step(self, batch, batch_idx):
        x_reconstruct, z_means, z_logvars = self(batch["central"])

        loss = self.loss(x_reconstruct, z_means, z_logvars, batch["central"])

        self.log("train_loss", loss, batch_size=len(batch["central"]))
        return loss

    def validation_step(self, batch, batch_idx):
        x_reconstruct, z_means, z_logvars = self(batch["central"])

        loss = self.loss(x_reconstruct, z_means, z_logvars, batch["central"])

        self.log("val_loss", loss, batch_size=len(batch["central"]))
        return loss

    def predict_step(self, batch):
        z = self.encoder(batch["central"])
        return self.to_mu(z), batch["0/taxon"]