import torch
import h5torch
import numpy as np
from maldi_nn.utils.drug import (
    SmilesToIndex,
    SmilesToECFP,
    SmilesToAlphabet,
    SmilesToDEEPSmilesAlphabet,
    SmilesToSelfiesAlphabet,
    SmilesToImage,
    SmilesToLINGOKernel,
)
import re
from maldi_nn.spectrum import SpectrumObject
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
import pandas as pd


def spectra_to_torch(spectra):
    spectra = [s.torch() for s in spectra]
    return {
        "mz": torch.stack([s["mz"] for s in spectra]),
        "intensity": torch.stack([s["intensity"] for s in spectra]),
    }


class MALDITOFDataset(h5torch.Dataset):
    def __init__(self, path, subset=None, preprocessor=None):
        super().__init__(path, subset=subset, sample_processor=self.sample_processor)
        self.preprocessor = preprocessor

    @staticmethod
    def create(path, spectrumobjects, labels, binned=False, **objects):
        f = h5torch.File(path, "w")

        f.register(labels, "central")

        if binned:
            intensities = np.stack([s.intensity for s in spectrumobjects])
            mz = spectrumobjects[0].mz
            f.register(intensities, axis=0, name="intensity")
            f.register(mz, axis="unstructured", name="mz")
        else:
            intensities = [s.intensity for s in spectrumobjects]
            mzs = [s.mz for s in spectrumobjects]
            f.register(intensities, axis=0, name="intensity", mode="vlen")
            f.register(mzs, axis=0, name="mz", mode="vlen")

        for k, v in objects.items():
            f.register(v, axis=0, name=k)

    def sample_processor(self, f, sample):
        spectrum = SpectrumObject(
            mz=(sample["0/mz"] if "0/mz" in sample else f["unstructured/mz"][:]),
            intensity=sample["0/intensity"],
        )

        if self.preprocessor is not None:
            spectrum = self.preprocessor(spectrum)

        spectrum = {
            "intensity": torch.tensor(spectrum.intensity).float(),
            "mz": torch.tensor(spectrum.mz),
        }

        sample = {k: v for k, v in sample.items() if k not in ["0/intensity", "0/mz"]}
        return sample | spectrum


class MaldiDataModule(LightningDataModule):
    def __init__(self, batch_size=16, n_workers=4):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=batch_collater,
        )


class DRIAMSAMRDataModule(MaldiDataModule):
    def __init__(
        self,
        path,
        drug_encoder="onehot",
        drug_encoder_args={},  # only for ecfp and image
        batch_size=16,
        n_workers=4,
        preprocessor=None,
        min_spectrum_len=None,
        in_memory=False,
    ):
        super().__init__(batch_size=batch_size, n_workers=n_workers)
        self.path = path

        smiles_list = sorted(
            list(set(list(h5torch.File(path)["1/drug_smiles"][:].astype(str))))
        )
        if drug_encoder == "onehot":
            self.drug_encoder = SmilesToIndex(smiles_list)
        elif drug_encoder == "ecfp":
            self.drug_encoder = SmilesToECFP(smiles_list, **drug_encoder_args)
        elif drug_encoder == "img":
            self.drug_encoder = SmilesToImage(smiles_list, **drug_encoder_args)
        elif drug_encoder == "kernel":
            f = h5torch.File(path)
            trainmatcher = np.vectorize(lambda x: bool(re.match("A_train", x)))
            train_indices = np.where(
                trainmatcher(f["unstructured/split"][:].astype(str))
            )[0]
            cols = np.unique(f["central/indices"][:][1][train_indices])
            smiles_list = sorted(
                list(set(list(f["1/drug_smiles"][:][cols].astype(str))))
            )
            f.close()
            self.drug_encoder = SmilesToLINGOKernel(smiles_list)

        elif drug_encoder in ["cnn", "trf", "gru"]:
            if drug_encoder_args["alphabet"] == "smiles":
                self.drug_encoder = SmilesToAlphabet(smiles_list)
            if drug_encoder_args["alphabet"] == "deepsmiles":
                self.drug_encoder = SmilesToDEEPSmilesAlphabet(smiles_list)
            if drug_encoder_args["alphabet"] == "selfies":
                self.drug_encoder = SmilesToSelfiesAlphabet(smiles_list)

        self.min_spectrum_len = min_spectrum_len
        self.preprocessor = preprocessor
        self.in_memory = in_memory

    def setup(self, stage):
        f = h5torch.File(self.path)
        trainmatcher = np.vectorize(lambda x: bool(re.match("A_train", x)))
        train_indices = np.where(trainmatcher(f["unstructured/split"][:].astype(str)))[
            0
        ]
        valmatcher = np.vectorize(lambda x: bool(re.match("A_val", x)))
        val_indices = np.where(valmatcher(f["unstructured/split"][:].astype(str)))[0]
        testmatcher = np.vectorize(lambda x: bool(re.match("A_test", x)))
        test_indices = np.where(testmatcher(f["unstructured/split"][:].astype(str)))[0]

        if self.min_spectrum_len is not None:
            lens_all = np.array([len(tt) for tt in f["0/intensity"][:]])

            lens = lens_all[f["central"]["indices"][0][train_indices]]
            train_indices = train_indices[lens > self.min_spectrum_len]

            lens = lens_all[f["central"]["indices"][0][val_indices]]
            val_indices = val_indices[lens > self.min_spectrum_len]

            lens = lens_all[f["central"]["indices"][0][test_indices]]
            test_indices = test_indices[lens > self.min_spectrum_len]

        if self.in_memory:
            f = f.to_dict()

        self.train = h5torch.Dataset(
            f,
            sampling="coo",
            subset=train_indices,
            sample_processor=self.processor,
        )

        self.val = h5torch.Dataset(
            f,
            sampling="coo",
            subset=val_indices,
            sample_processor=self.processor,
        )

        self.test = h5torch.Dataset(
            f,
            sampling="coo",
            subset=test_indices,
            sample_processor=self.processor,
        )

    def processor(self, f, sample):
        spectrum = SpectrumObject(
            mz=(sample["0/mz"] if "0/mz" in sample else f["unstructured/mz"][:]),
            intensity=sample["0/intensity"],
        )

        if self.preprocessor is not None:
            spectrum = self.preprocessor(spectrum)
        spectrum = {
            "intensity": torch.tensor(spectrum.intensity).float(),
            "mz": torch.tensor(spectrum.mz),
        }

        return (
            spectrum
            | {"label": int(sample["central"].astype(str) != "S")}
            | {"drug": self.drug_encoder(sample["1/drug_smiles"].astype(str))}
            | {"species": sample["0/species"]}
            | {
                "loc": sample["0/loc"].astype(str),
                "drug_name": sample["1/drug_names"].astype(str),
            }
        )


class DRIAMSSpectrumDataModule(MaldiDataModule):
    def __init__(
        self,
        path,
        batch_size=512,
        n_workers=4,
        preprocessor=None,
        min_spectrum_len=None,
        in_memory=True,
        exclude_nans=False,
    ):
        super().__init__(batch_size=batch_size, n_workers=n_workers)
        self.path = path

        self.min_spectrum_len = min_spectrum_len

        f = h5torch.File(self.path)
        trainmatcher = np.vectorize(lambda x: bool(re.match("A_train", x)))
        self.train_indices = np.where(
            trainmatcher(f["unstructured/split"][:].astype(str))
        )[0]
        valmatcher = np.vectorize(lambda x: bool(re.match("A_val", x)))
        self.val_indices = np.where(valmatcher(f["unstructured/split"][:].astype(str)))[
            0
        ]
        testmatcher = np.vectorize(lambda x: bool(re.match("A_test", x)))
        self.test_indices = np.where(
            testmatcher(f["unstructured/split"][:].astype(str))
        )[0]

        if self.min_spectrum_len is not None:
            lens = np.array([len(tt) for tt in f["0/intensity"][:][self.train_indices]])
            self.train_indices = self.train_indices[lens > self.min_spectrum_len]

            lens = np.array([len(tt) for tt in f["0/intensity"][:][self.val_indices]])
            self.val_indices = self.val_indices[lens > self.min_spectrum_len]

            lens = np.array([len(tt) for tt in f["0/intensity"][:][self.test_indices]])
            self.test_indices = self.test_indices[lens > self.min_spectrum_len]

        self.preprocessor = preprocessor
        self.in_memory = in_memory

        species_labels = f["unstructured/species_labels"][:]

        species_train = f["central"][:][self.train_indices]
        species_val = f["central"][:][self.val_indices]
        species_test = f["central"][:][self.test_indices]

        species_mapping = {i: i for i in range(len(species_labels))}

        nan = np.where(species_labels == b"nan")[0].item()

        mix_labels = pd.Series(species_labels.astype(str)).str.startswith("MIX").values
        for i in np.where(mix_labels)[0]:
            species_mapping[i] = nan

        s, c = np.unique(species_train, return_counts=True)
        for i in s[c < 5]:
            species_mapping[i] = nan

        exclude_val = np.array(
            list(set(np.unique(species_val)).difference(set(np.unique(species_train))))
        )
        exclude_test = np.array(
            list(set(np.unique(species_test)).difference(set(np.unique(species_train))))
        )
        for i in exclude_val:
            species_mapping[i] = nan
        for i in exclude_test:
            species_mapping[i] = nan

        # not in train/val/test of hosp A (but existing in other hospitals):
        t = np.concatenate(
            [
                [species_mapping[f_] for f_ in species_train],
                [species_mapping[f_] for f_ in species_val],
                [species_mapping[f_] for f_ in species_test],
            ]
        )
        for i in set(species_mapping.values()).difference(np.unique(t)):
            species_mapping[i] = nan

        c = 0
        map_to_enum = {nan: nan}
        for k, v in species_mapping.items():
            if v not in map_to_enum:
                map_to_enum[v] = c
                c += 1

        self.species_mapping = {k: map_to_enum[v] for k, v in species_mapping.items()}

        if exclude_nans:
            train_nan = np.array([species_mapping[s] != nan for s in species_train])
            val_nan = np.array([species_mapping[s] != nan for s in species_val])
            test_nan = np.array([species_mapping[s] != nan for s in species_test])

            self.train_indices = self.train_indices[train_nan]
            self.val_indices = self.val_indices[val_nan]
            self.test_indices = self.test_indices[test_nan]

        self.n_species = len(
            np.unique(
                [self.species_mapping[f_] for f_ in f["central"][:][self.train_indices]]
            )
        )

        f.close()

    def setup(self, stage):
        f = h5torch.File(self.path)
        if self.min_spectrum_len is not None:
            lens = np.array([len(tt) for tt in f["0/intensity"][:][self.train_indices]])
            self.train_indices = self.train_indices[lens > self.min_spectrum_len]

            lens = np.array([len(tt) for tt in f["0/intensity"][:][self.val_indices]])
            self.val_indices = self.val_indices[lens > self.min_spectrum_len]

            lens = np.array([len(tt) for tt in f["0/intensity"][:][self.test_indices]])
            self.test_indices = self.test_indices[lens > self.min_spectrum_len]

        if self.in_memory:
            f = f.to_dict()

        self.train = h5torch.Dataset(
            f, subset=self.train_indices, sample_processor=self.processor
        )

        self.val = h5torch.Dataset(
            f, subset=self.val_indices, sample_processor=self.processor
        )

        self.test = h5torch.Dataset(
            f, subset=self.test_indices, sample_processor=self.processor
        )

    def processor(self, f, sample):
        spectrum = SpectrumObject(
            mz=(sample["0/mz"] if "0/mz" in sample else f["unstructured/mz"][:]),
            intensity=sample["0/intensity"],
        )

        if self.preprocessor is not None:
            spectrum = self.preprocessor(spectrum)

        spectrum = {
            "intensity": torch.tensor(spectrum.intensity).float(),
            "mz": torch.tensor(spectrum.mz),
        }

        return (
            spectrum
            | {
                "species": (
                    sample["central"]
                    if self.species_mapping is None
                    else self.species_mapping[sample["central"]]
                )
            }
            | {"loc": sample["0/loc"].astype(str)}
        )


class SpeciesClfDataModule(MaldiDataModule):
    def __init__(
        self,
        path,
        batch_size=512,
        n_workers=4,
        preprocessor=None,
        in_memory=True,
    ):
        super().__init__(batch_size=batch_size, n_workers=n_workers)
        self.path = path
        self.preprocessor = preprocessor
        self.in_memory = in_memory

    def setup(self, stage):
        f = h5torch.File(self.path)

        if self.in_memory:
            f = f.to_dict()

        self.train = MALDITOFDataset(
            f, subset=("unstructured/split", "train"), preprocessor=self.preprocessor
        )

        self.val = MALDITOFDataset(
            f, subset=("unstructured/split", "val"), preprocessor=self.preprocessor
        )

        self.test = MALDITOFDataset(
            f, subset=("unstructured/split", "test"), preprocessor=self.preprocessor
        )

        self.n_species = len(f["unstructured/species_labels"])


def batch_collater(batch):
    batch_collated = {}
    keys = list(batch[0])
    for k in keys:
        v = [b[k] for b in batch]
        if isinstance(v[0], str):
            batch_collated[k] = v
        elif isinstance(v[0], (int, np.int64)):
            batch_collated[k] = torch.tensor(v)
        elif isinstance(v[0], np.ndarray):
            if len({t.shape for t in v}) == 1:
                batch_collated[k] = torch.tensor(np.array(v))
            else:
                batch_collated[k] = pad_sequence(
                    [torch.tensor(t) for t in v], batch_first=True, padding_value=-1
                )
        elif torch.is_tensor(v[0]):
            if len({t.shape for t in v}) == 1:
                batch_collated[k] = torch.stack(v)
            else:
                if v[0].dtype == torch.bool:
                    batch_collated[k] = pad_sequence(
                        v, batch_first=True, padding_value=False
                    )
                else:
                    batch_collated[k] = pad_sequence(
                        v, batch_first=True, padding_value=-1
                    )
    return batch_collated
