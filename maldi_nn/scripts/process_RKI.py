from maldi_nn.utils.data import *
import sys
from importlib.resources import files
import h5torch
import json
import os
import pandas as pd
import numpy as np
from maldi_nn.spectrum import *
import shutil

def RKI_raw_to_h5torch(RKI_ROOT, outfile):
    data_path = files("maldi_nn.utils").joinpath("RKI_split.json")
    split = json.load(open(data_path))

    species = []
    subspecies = []
    spectra = []
    loc = []
    for l in os.walk(RKI_ROOT):
        root, subfolders, files_ = l
        if "fid" in files_ and "acqu" in files_:
            r_ = root.split("/")
            species.append(r_[-6])
            subspecies.append(r_[-5])
            loc.append(root.lstrip(RKI_ROOT))
            s = SpectrumObject.from_bruker(
                os.path.join(root, "acqu"), os.path.join(root, "fid")
            )
            spectra.append(s)

    species_map = {k: v for v, k in enumerate(np.unique(species))}
    species_map_inv = {v: k for k, v in species_map.items()}
    species_labels = np.array([species_map_inv[i] for i in range(len(species_map_inv))])
    species = np.array([species_map[i] for i in species])

    spectrum_split = np.array([split[l] for l in loc])

    f = h5torch.File(outfile, "w")
    f.register(species, "central")
    f.register(np.array(loc).astype(bytes), 0, name="loc")
    f.register(species_labels.astype(bytes), "unstructured", name="species_labels")
    f.register(spectrum_split.astype(bytes), "unstructured", name="split")

    ints = [s.intensity for s in spectra]
    mzs = [s.mz for s in spectra]

    f.register(ints, 0, name="intensity", mode="vlen")
    f.register(mzs, 0, name="mz", mode="vlen")
    f.close()


def RKI_raw_to_binned(rawfile, processed_file):
    binner = SequentialPreprocessor(
        VarStabilizer(method="sqrt"),
        Smoother(halfwindow=10),
        BaselineCorrecter(method="SNIP", snip_n_iter=20),
        Trimmer(),
        Binner(),
        Normalizer(sum=1),
    )
    shutil.copy(rawfile, processed_file)
    file = h5torch.File(processed_file, "a")
    len_ = file["0/mz"].shape[0]
    ints = []
    for i in range(len_):
        mz = file["0/mz"][i]
        intensity = file["0/intensity"][i]
        s = SpectrumObject(mz=mz, intensity=intensity)
        ints.append(binner(s).intensity)
        if (i + 1) % 1000 == 0:
            print(i, end=" ", flush=True)

    del file["0/mz"]
    del file["0/intensity"]

    file.register(np.stack(ints), 0, name="intensity")
    file.register(binner(s).mz, "unstructured", name="mz")

    file.close()
    return None


def RKI_raw_to_peaks(rawfile, processed_file):
    peakdetector = SequentialPreprocessor(
        VarStabilizer(method="sqrt"),
        Smoother(halfwindow=10),
        BaselineCorrecter(method="SNIP", snip_n_iter=20),
        Trimmer(),
        PersistenceTransformer(extract_nonzero=True),
        Normalizer(sum=1),
        PeakFilter(max_number=2048),
    )
    shutil.copy(rawfile, processed_file)
    file = h5torch.File(processed_file, "a")
    len_ = file["0/mz"].shape[0]
    ints = []
    mzs = []
    for i in range(len_):
        mz = file["0/mz"][i]
        intensity = file["0/intensity"][i]
        s = SpectrumObject(mz=mz, intensity=intensity)
        s = peakdetector(s)
        ints.append(s.intensity)
        mzs.append(s.mz)
        if (i + 1) % 1000 == 0:
            print(i, end=" ", flush=True)

    del file["0/mz"]
    del file["0/intensity"]
    file.register(ints, 0, name="intensity", mode="vlen")
    file.register(mzs, 0, name="mz", mode="vlen")

    file.close()
    return None


def main():
    RKI_root = str(sys.argv[1])
    spectraraw = str(sys.argv[2])
    spectrabin = str(sys.argv[3])
    spectrapks = str(sys.argv[4])

    RKI_raw_to_h5torch(RKI_root, spectraraw)
    RKI_raw_to_binned(spectraraw, spectrabin)
    RKI_raw_to_peaks(spectraraw, spectrapks)

if __name__ == "__main__":
    main()