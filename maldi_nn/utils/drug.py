from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import AllChem
import numpy as np
import deepsmiles
import selfies
import re
import torch
import maldi_nn.nn as maldinn
import torch.nn as nn
from bio_attention.attention import GLU, TransformerEncoder, Aggregator
from bio_attention.embed import DiscreteEmbedding


class SmilesEncoder:
    def __init__(self):
        self.vocab = {}
        self.embeds = []
        self.alphabet_dict = {}

    def __call__(self, smiles_string):
        return (
            self.vocab[smiles_string]
            if smiles_string in self.vocab
            else self.embed(smiles_string)
        )

    def embed(self, smiles_string):
        return NotImplementedError

    def create_vocab(self, smiles_list):
        return {smiles: self.embed(smiles) for smiles in smiles_list}


class SmilesToIndex(SmilesEncoder):
    def __init__(self, smiles_list):
        super().__init__()
        self.vocab = self.create_vocab(smiles_list)

    def embed(self, smiles_string):
        raise NotImplementedError("Can't embed new SMILES strings.")

    def create_vocab(self, smiles_list):
        return {smiles: ix for ix, smiles in enumerate(smiles_list)}


class SmilesToECFP(SmilesEncoder):
    def __init__(self, smiles_list, bits=1024, diameter=4):
        super().__init__()

        self.bits = bits
        self.rad = int(diameter / 2)

        self.vocab = self.create_vocab(smiles_list)

    def embed(self, smiles_string):
        molecule = Chem.MolFromSmiles(smiles_string)
        return np.array(
            AllChem.GetMorganFingerprintAsBitVect(molecule, self.rad, nBits=self.bits)
        ).astype(np.float32)


class SmilesToAlphabet(SmilesEncoder):
    def __init__(self, smiles_list):
        super().__init__()

        alphabet = set([c for smiles in smiles_list for c in smiles])
        alphabet = sorted(alphabet)
        self.alphabet_dict = {a: i for i, a in enumerate(alphabet)}
        self.alphabet_dict["[UNK]"] = len(self.alphabet_dict)
        self.onehot_selector = np.eye(len(self.alphabet_dict))

        self.vocab = self.create_vocab(smiles_list)

    def embed(self, smiles_string):
        smiles_ints = self.string2onehot(smiles_string)
        return self.onehot_selector[smiles_ints].astype(np.float32)

    def string2onehot(self, string):
        return np.array(
            [
                self.alphabet_dict[c]
                if c in self.alphabet_dict
                else self.alphabet_dict["[UNK]"]
                for c in string
            ]
        )


class SmilesToDEEPSmilesAlphabet(SmilesEncoder):
    def __init__(self, smiles_list):
        super().__init__()
        self.converter = deepsmiles.Converter(rings=True, branches=True)

        alphabet = set(
            [c for smiles in smiles_list for c in self.smiles_to_deepsmiles(smiles)]
        )
        alphabet = sorted(alphabet)
        self.alphabet_dict = {a: i for i, a in enumerate(alphabet)}
        self.alphabet_dict["[UNK]"] = len(self.alphabet_dict)
        self.onehot_selector = np.eye(len(self.alphabet_dict))

        self.vocab = self.create_vocab(smiles_list)

    def embed(self, smiles_string):
        smiles_ints = self.string2onehot(smiles_string)
        return self.onehot_selector[smiles_ints].astype(np.float32)

    def string2onehot(self, string):
        return np.array(
            [
                self.alphabet_dict[c]
                if c in self.alphabet_dict
                else self.alphabet_dict["[UNK]"]
                for c in self.smiles_to_deepsmiles(string)
            ]
        )

    def smiles_to_deepsmiles(self, smiles):
        return self.converter.encode(smiles)


class SmilesToSelfiesAlphabet(SmilesEncoder):
    def __init__(self, smiles_list):
        super().__init__()

        alphabet = selfies.get_alphabet_from_selfies(
            [selfies.encoder(smiles) for smiles in smiles_list]
        )
        alphabet = sorted(alphabet)
        self.alphabet_dict = {a: i for i, a in enumerate(alphabet)}
        self.alphabet_dict["."] = len(self.alphabet_dict)
        self.alphabet_dict["[UNK]"] = len(self.alphabet_dict)
        self.onehot_selector = np.eye(len(self.alphabet_dict))

        self.vocab = self.create_vocab(smiles_list)

    def embed(self, smiles_string):
        smiles_ints = self.string2onehot(smiles_string)
        return self.onehot_selector[smiles_ints].astype(np.float32)

    def string2onehot(self, string):
        return np.array(
            [
                self.alphabet_dict[c]
                if c in self.alphabet_dict
                else self.alphabet_dict["[UNK]"]
                for c in selfies.split_selfies(self.smiles_to_selfies(string))
            ]
        )

    def smiles_to_selfies(self, smiles):
        return selfies.encoder(smiles)


class SmilesToImage(SmilesEncoder):
    def __init__(self, smiles_list, size=(150, 150), normalize=True, inverse=False):
        super().__init__()
        self.size = size
        self.normalize = normalize
        self.inverse = inverse
        self.vocab = self.create_vocab(smiles_list)

    def embed(self, smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)
        embed = np.array(MolToImage(mol, size=self.size))
        if self.inverse:
            embed = 255 - embed
        if self.normalize:
            embed = embed / 255

        return embed.transpose(2, 0, 1).astype(np.float32)


class SmilesToLINGOKernel(SmilesEncoder):
    def __init__(self, smiles_list):
        super().__init__()
        embedded = [self.get_n_fourmers(smiles) for smiles in smiles_list]

        kernel = np.zeros((len(embedded), len(embedded)))
        for i in range(len(embedded)):
            for j in range(i, len(embedded)):
                sim = self.LINGOsim(embedded[i], embedded[j])
                kernel[i, j] = sim
                kernel[j, i] = sim

        self.vocab = {smiles: kernel[ix, :] for ix, smiles in enumerate(smiles_list)}
        self.embeds = embedded

    def embed(self, smiles_string):
        embedded = self.get_n_fourmers(smiles_string)
        return np.array([self.LINGOsim(embedded, n) for n in self.embeds])

    def get_n_fourmers(self, smiles):
        smiles = re.sub("\d", "o", smiles)
        fourmers = [smiles[s : s + 4] for s in range(len(smiles) - 3)]
        return {s: c for s, c in zip(*np.unique(fourmers, return_counts=True))}

    def LINGOsim(self, s1, s2):
        l_ = set.union(*[set(s1.keys()), set(s2.keys())])
        return sum(
            [
                1
                - abs(self.take(s1, l) - self.take(s2, l))
                / abs(self.take(s1, l) + self.take(s2, l))
                for l in l_
            ]
        ) / len(l_)

    @staticmethod
    def take(dict_, elem):
        return dict_[elem] if elem in dict_ else 0


class DrugOneHotEmbedding(nn.Module):
    def __init__(self, num_drugs, dim):
        super().__init__()
        self.embed = nn.Embedding(num_drugs, dim)

    def forward(self, drug):
        return self.embed(drug.long())


class DrugMLP(maldinn.MLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, drug):
        return self.net(drug)


class DrugGRU(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, bidirectional=True, out_head=64):
        super().__init__()
        self.GRU = nn.GRU(
            vocab_size,
            hidden_dim,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_head = nn.Linear(hidden_dim, out_head)

    def forward(self, drug):
        mask = (drug != -1).all(-1)
        x = nn.utils.rnn.pack_padded_sequence(
            drug, mask.sum(1).tolist(), batch_first=True, enforce_sorted=False
        )
        _, out = self.GRU(x)

        return self.output_head(out.mean(0))


class MaskedConv1dWrapper(nn.Module):
    def __init__(self, ConvModule):
        """
        Works only for Conv1ds that have the same input shape as output shape.
        """
        assert isinstance(ConvModule, nn.Conv1d)
        super().__init__()
        self.conv = ConvModule

    def forward(self, x, mask=None):
        """
        x = (B, C, L)
        mask = (B, L)
        """
        if mask is not None:
            return self.conv(x * mask.unsqueeze(1))
        else:
            return self.conv(x)


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        depthwise=False,
        glu_ff=True,
        dropout=0.2,
        activation="gelu",
    ):
        super().__init__()
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "swish":
            act = nn.SiLU()

        self.conv = MaskedConv1dWrapper(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=(dim if depthwise else 1),
            )
        )

        self.norm = nn.Sequential(
            maldinn.Permute(0, 2, 1),
            nn.LayerNorm(dim),
            maldinn.Permute(0, 2, 1),
        )

        self.prenorm = nn.Sequential(
            maldinn.Permute(0, 2, 1),
            nn.LayerNorm(dim),
            maldinn.Permute(0, 2, 1),
        )

        project_in = (
            nn.Sequential(nn.Linear(dim, 4 * dim), act)
            if not glu_ff
            else GLU(dim, 4 * dim, act)
        )

        self.pointwise_net = nn.Sequential(
            maldinn.Permute(0, 2, 1),
            project_in,
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            maldinn.Permute(0, 2, 1),
        )

    def forward(self, x, mask=None):
        z = self.conv(self.prenorm(x), mask=mask) + x
        z = self.pointwise_net(self.norm(z)) + z
        return z


class DrugCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=64,
        depth=4,
        kernel_size=5,
        dropout=0.2,
        out_head=64,
    ):
        super().__init__()

        self.stem = nn.Sequential(nn.Linear(vocab_size, dim), maldinn.Permute(0, 2, 1))

        body = []
        for _ in range(depth):
            body.append(
                Conv1DBlock(
                    dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    depthwise=False,
                    glu_ff=True,
                    activation="gelu",
                )
            )
        self.body = nn.ModuleList(body)
        self.output_head = nn.Linear(dim, out_head)

    def forward(self, drug):
        mask = (drug != -1).all(-1)
        z = self.stem(drug)
        for layer in self.body:
            z = layer(z, mask=mask)

        return self.output_head(z.max(2).values)


class DrugTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=128,
        depth=6,
        n_head=8,
        dropout=0.2,
        out_head=64,
    ):
        super().__init__()

        self.embed = DiscreteEmbedding(vocab_size, dim, cls=True)
        self.body = TransformerEncoder(
            depth=depth,
            dim=dim,
            nh=n_head,
            attention_args={"dropout": dropout},
            plugintype="sinusoidal",
            plugin_args={"dim": dim},
            only_apply_plugin_at_first=True,
            dropout=dropout,
            glu_ff=True,
            activation="gelu",
        )

        self.agg = Aggregator(method="cls")
        self.output_head = nn.Linear(dim, out_head)

    def forward(self, drug):
        mask = (drug != -1).all(-1)
        drug = torch.argmax(drug, dim=2).to(self.embed.embedder.weight.dtype)
        drug[~mask] = torch.nan

        x = self.embed(drug)

        z = self.body(x, mask=mask)

        return self.output_head(self.agg(z))


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=5,
        dropout=0.2,
        depthwise=False,
        glu_ff=True,
        activation="gelu",
    ):
        super().__init__()

        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "swish":
            act = nn.SiLU()

        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=(dim if depthwise else 1),
        )

        self.norm = nn.Sequential(
            maldinn.Permute(0, 2, 3, 1),
            nn.LayerNorm(dim),
            maldinn.Permute(0, 3, 1, 2),
        )

        project_in = (
            nn.Sequential(nn.Linear(dim, 4 * dim), act)
            if not glu_ff
            else GLU(dim, 4 * dim, act)
        )

        self.pointwise_net = nn.Sequential(
            maldinn.Permute(0, 2, 3, 1),
            project_in,
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            maldinn.Permute(0, 3, 1, 2),
        )

        self.prenorm = nn.Sequential(
            maldinn.Permute(0, 2, 3, 1),
            nn.LayerNorm(dim),
            maldinn.Permute(0, 3, 1, 2),
        )

    def forward(self, x):
        z = self.conv(self.prenorm(x)) + x
        z = self.pointwise_net(self.norm(z)) + z
        return z


class DrugImageCNN(nn.Module):
    def __init__(
        self,
        stem_downsample=2,
        kernel_size=5,
        hidden_dim=32,
        depth=2,
        dropout=0.2,
        out_head=64,
    ):
        super().__init__()

        body = [nn.Conv2d(3, hidden_dim, stem_downsample, stride=stem_downsample)]

        for _ in range(depth):
            body.append(
                Conv2DBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    glu_ff=True,
                    activation="gelu",
                    depthwise=False,
                )
            )

        self.body = nn.Sequential(*body)

        self.output_head = nn.Linear(hidden_dim, out_head)

    def forward(self, drug):
        z = self.body(drug)
        return self.output_head(z.max(-1).values.max(-1).values)
