import torch
import h5torch
import numpy as np

def spectra_to_torch(spectra):
    spectra = [s.torch() for s in spectra]
    return {
        "mz": torch.stack([s["mz"] for s in spectra]),
        "intensity": torch.stack([s["intensity"] for s in spectra])
        }

class MALDITOFDataset(h5torch.Dataset):
    def __init__(self, path, subset = None):
        super().__init__(
            path, subset=subset, sample_processor=self.sample_processor
        )

    @staticmethod
    def create(path, spectrumobjects, binned = False, **objects):
        f = h5torch.File(path, "w")
        if binned:
            intensities = np.stack([s.intensity for s in spectrumobjects])
            mz = spectrumobjects[0].mz
            f.register(intensities, "central")
            f.register(mz, axis = 1, name = "mz")
        else:
            intensities = [s.intensity for s in spectrumobjects]
            mzs = [s.mz for s in spectrumobjects]
            f.register(intensities, "central", mode = "vlen")
            f.register(mzs, axis = 0, name = "mz", mode = "vlen")
        
        for k, v in objects.items():
            f.register(v, axis = 0, name = k)

    @staticmethod
    def sample_processor(f, sample):
        intensity = sample["central"]
        if "0/mz" in sample:
            mz = sample["0/mz"]
        else:
            mz = f["1/mz"][:]
        sample = {k[2:] : v for k, v in sample.items() if k not in ["central", "0/mz"]}
        return sample | {"intensity" : intensity, "mz" : mz}