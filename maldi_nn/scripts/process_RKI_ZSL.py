import os
import requests
import numpy as np
import pandas as pd
from Bio import SeqIO
import h5torch
from maldi_nn.spectrum import *
import re
import lpsn
from tqdm import tqdm
from Bio import Entrez
import sys
import pandas as pd
import numpy as np
import h5torch
import sys
from importlib.resources import files
import shutil

ALIGN_TOKEN_MAPPING = {'-': 0,
 '.': 1,
 'A': 2,
 'B': 3,
 'C': 4,
 'D': 5,
 'G': 6,
 'H': 7,
 'K': 8,
 'M': 9,
 'N': 10,
 'R': 11,
 'S': 12,
 'U': 13,
 'V': 14,
 'W': 15,
 'Y': 16
}


def RKI_ZSL_raw_to_h5torch(RKI_root, SILVA_fasta, lpsn_email, lpsn_pw, outfile):
    # Read in all fasta sequences
    fasta_sequences = list(SeqIO.parse(open(SILVA_fasta),'fasta'))

    # list all files in the RKI data set
    t = []
    for l in os.walk(RKI_root):
        root, subfolders, files_ = l
        if "fid" in files_ and "acqu" in files_:
            r_ = root.split("/")
            genus = r_[-7]
            species = r_[-6].removeprefix(r_[-7]).strip()
            subspecies = r_[-5].removeprefix(r_[-6]).strip()
            t.append([genus, species, subspecies])

    # Unique strains in dataset
    uniq_strains = pd.DataFrame(t).drop_duplicates(keep='first').values


    aas = []
    ncbi_ids = []

    
    Entrez.email = 'A.N.Other@example.com'

    client = lpsn.LpsnClient(lpsn_email, lpsn_pw) # Add your login details from lpsn bacterio net here.

    def lookup(str_): # Straininfo lookup helper function
        req = requests.get('https://api.straininfo.dsmz.de/v1/search/culture/str_des/%s' % (str_)).json()
        return requests.get('https://api.straininfo.dsmz.de/v1/data/culture/max/%s' % (req[0])).json()


    for ix in tqdm(range(len(uniq_strains))): # iterate over unique strains in detaset
        # partim one: lookup strain name in StrainInfo
        key = uniq_strains[ix, -1].replace(" ", "%20")
        try:
            res = lookup(key) # look up strain in straininfo
        except:
            # If this fails:
            # Some strains are called e.g. Genus species DSM xxx (B ...), this repeats the search without the (B ...)
            key = re.sub("\(.*\)", "", uniq_strains[ix, -1]).rstrip().replace(" ", "%20") 
            try:
                res = lookup(key)
            except:
                try:
                    # If that does not work, some strain are called e.g. Genus species DSM xxx ATCC xxx,
                    # this repeats the search with only the DSM part.
                    key = re.search("DSM \d*", uniq_strains[ix, -1]).group(0).replace(" ", "%20")
                    res = lookup(key)
                except:
                    aas.append("Culture not found")
                    continue


        # get the strain names from the matched culture in straininfo
        match_list = [res[0]["culture"]["strain_number"]]
        if "relation" in res[0]["culture"]:
            match_list += res[0]["culture"]["relation"]

        # get the genus and species name from the matched culture in straininfo
        try:
            taxon = res[0]["culture"]["taxon"]["name"]
        except:
            aas.append("No culture id in found record")
            continue


        # only if the genus, species and culture name correspond to the one we gave (i.e. no false match), proceed
        if not (any([m in key.replace("%20", " ") for m in match_list]) and any([l in uniq_strains[ix] for l in taxon.split(" ")])):
            aas.append("Culture name found but matched to wrong ID in straininfo.")
            continue

        # try to look up sequences in the straininfo match
        try:
            for l in sorted(res[0]["strain"]["sequence"], key= lambda x:-x["year"]):
                aa = [str(t.seq) for t in fasta_sequences if l["accession_number"] in t.description]
                if len(aa) > 0:
                    break
            if len(aa) == 0:
                raise ValueError
            aas.append(aa)
            ncbi_ids.append(res[0]["culture"]["taxon"])
            
        except:
            # if no sequence in the straininfo match, look up the strain in LPSN database.
            try:
                count = client.search(id=res[0]["culture"]["taxon"]["lpsn"])
                if count > 0:
                    entry = list(client.retrieve())[0]
                    if any([m in key.replace("%20", " ") for m in entry["type_strain_names"]]):
                        for m in entry["molecules"]:
                            if m["kind"] == "16S rRNA gene":
                                aa = [str(t.seq) for t in fasta_sequences if m["identifier"] in t.description]
                                if len(aa) > 0:
                                    break
                        if len(aa) == 0:
                            raise ValueError
                        aas.append(aa)
                        ncbi_ids.append(res[0]["culture"]["taxon"])
                    else:
                        aas.append("No matching seq found in lpsn")
                else:
                    aas.append("No matching seq found in lpsn")
            except:
                aas.append("No matching seq found")


    mapper = {
        "Bacillus" : "Bacillaceae",
        "Pseudomonas" : "Pseudomonadaceae",
        "Staphylococcus" : "Staphylococcaceae",
        "Yersinia" : "Yersiniaceae",
        "Francisella" : "Francisellaceae",
    }

    def ncbi_id_to_fam(id):
        attempts = 0
        while attempts < 5:
            try:
                handle = Entrez.efetch(db='taxonomy', id = id, rettype="xml")
                records = handle.read()
                lineage = records.decode("utf-8").split("Lineage")[1][1:-2].split("; ")
                break
            except:
                attempts += 1
        return [l for l in lineage if l.endswith("ae")][0]

    name_to_seq = {}
    c = 0
    for i in range(len(uniq_strains)):
        if len(aas[i][0]) > 1:
            if "ncbi" in ncbi_ids[c]:
                familyname = ncbi_id_to_fam(ncbi_ids[c]["ncbi"])
            else:
                familyname = mapper[ncbi_ids[c]["name"].split(" ")[0]]
            c += 1
            name = familyname + ";" + ";".join(uniq_strains[i])
            name_to_seq[name] = aas[i][0]
            
    name_to_ix = {name : ix for ix, name in enumerate(name_to_seq)}
    ix_to_name = {v : k for k, v in name_to_ix.items()}


    species_names = [";".join(ix_to_name[i].split(";")[:-1]) for i in range(len(ix_to_name))]
    species_ix_names = []
    c = -1
    strain_to_species_ix = []
    for species_name in species_names:
        if species_name not in species_ix_names:
            species_ix_names.append(species_name)
            c += 1
        strain_to_species_ix.append(c)


    genus_names = [";".join(ix_to_name[i].split(";")[:-2]) for i in range(len(ix_to_name))]
    genus_ix_names = []
    c = -1
    strain_to_genus_ix = []
    for genus_name in genus_names:
        if genus_name not in genus_ix_names:
            genus_ix_names.append(genus_name)
            c += 1
        strain_to_genus_ix.append(c)

    strain_to_species_ix = np.array(strain_to_species_ix)
    strain_to_genus_ix = np.array(strain_to_genus_ix)

    strain_names = np.array([ix_to_name[i] for i in range(len(ix_to_name))])
    strain_seq = np.array([name_to_seq[i] for i in strain_names])

    strain_seq_encoded = []
    for seq in strain_seq:
        strain_seq_encoded.append([ALIGN_TOKEN_MAPPING[nt] for nt in seq])
    strain_seq_encoded = np.array(strain_seq_encoded)

    spectra = []
    strain_ix = []
    loc = []
    # list all files in the RKI data set
    name_up_to_genus_to_ix = {";".join(name.split(";")[1:]) : ix for ix, name in enumerate(name_to_seq)}
    for l in os.walk(RKI_root):
        root, subfolders, files_ = l
        if "fid" in files_ and "acqu" in files_:
            r_ = root.split("/")
            genus = r_[-7]
            species = r_[-6].removeprefix(r_[-7]).strip()
            subspecies = r_[-5].removeprefix(r_[-6]).strip()
            name = ";".join([genus, species, subspecies])
            if name in name_up_to_genus_to_ix:
                strain_ix.append(name_up_to_genus_to_ix[name])
                s = SpectrumObject.from_bruker(
                    os.path.join(root, "acqu"), os.path.join(root, "fid")
                )
                spectra.append(s)
                loc.append(root.removeprefix(RKI_root))

    ints = [s.intensity for s in spectra]
    mzs = [s.mz for s in spectra]
    species_ix = strain_to_species_ix[strain_ix]


    score_matrix = np.zeros((len(strain_ix), len(name_to_ix)), dtype=bool)
    score_matrix[np.arange(len(score_matrix)), strain_ix] = 1

    f = h5torch.File(outfile, "w")

    f.register(score_matrix, axis = "central", mode = "N-D", dtype_save = "bool", dtype_load = "int64")

    f.register(ints, 0, name="intensity", mode="vlen", dtype_save = "int32", dtype_load="int64")
    f.register(mzs, 0, name="mz", mode="vlen", dtype_save = "float32", dtype_load="float32")
    f.register(np.array(strain_ix), 0, name="strain_ix", mode="N-D", dtype_save = "int64", dtype_load="int64")
    f.register(np.array(species_ix), 0, name="species_ix", mode="N-D", dtype_save = "int64", dtype_load="int64")
    f.register(np.array(loc).astype(bytes), 0, name="loc", mode="N-D", dtype_save="bytes", dtype_load="str")

    f.register(np.array(strain_to_species_ix), 1, name="strain_to_species_ix", mode="N-D", dtype_save="int64", dtype_load="int64")
    f.register(np.array(strain_to_genus_ix), 1, name="strain_to_genus_ix", mode="N-D", dtype_save="int64", dtype_load="int64")
    f.register(np.array(strain_names), 1, name="strain_names", mode="N-D", dtype_save="bytes", dtype_load="str")
    f.register(np.array(strain_seq), 1, name="strain_seq", mode="N-D", dtype_save="bytes", dtype_load="str")
    f.register(np.array(strain_seq_encoded), 1, name="strain_seq_encoded", mode="N-D", dtype_save="int", dtype_load="int")

    f.register(np.array(species_ix_names), "unstructured", name="species_ix_to_name", mode="N-D", dtype_save="bytes", dtype_load="str")

    f.close()
    return None


def add_splits(outfile):
    f = h5torch.File(outfile, "a")

    for repeat in range(10):
        data_path = files("maldi_nn.utils.zsl_splits").joinpath("RKI_split_%s.txt" % repeat)
        indicator = np.loadtxt(data_path, dtype='str')
        f.register(indicator.astype(bytes), axis = 0, name = "split_%s" % repeat, mode = "N-D", dtype_save="bytes", dtype_load="str")
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

def main():
    RKI_root = str(sys.argv[1])
    SILVA_fasta = str(sys.argv[2])
    lpsn_email = str(sys.argv[3])
    lpsn_pw = str(sys.argv[4])
    outfile = str(sys.argv[5])
    outfile2 = str(sys.argv[6])

    RKI_ZSL_raw_to_h5torch(RKI_root, SILVA_fasta, lpsn_email, lpsn_pw, outfile)
    add_splits(outfile)
    RKI_raw_to_binned(outfile, outfile2)

if __name__ == "__main__":
    main()