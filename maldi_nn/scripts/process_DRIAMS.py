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

# Obtained from original DRIAMS publication Weis et al.
drug_to_class = {
    "5-Fluorocytosine": "antimycotic systemic",
    "Amikacin": "aminoglycoside",
    "Amoxicillin": "penicillins",
    "Amoxicillin + Clavulanic acid": "penicillins",
    "Amphotericin B": "antimycotic systemic",
    "Ampicillin": "penicillins",
    "Ampicillin + Amoxicillin": "penicillins",
    "Ampicillin-Sulbactam": "penicillins",
    "Anidulafungin": "antimycotic systemic",
    "Azithromycin": "macrolides",
    "Aztreonam": "other beta-lactam",
    "Bacitracin A": "other",
    "Caspofungin": "antimycotic systemic",
    "Cefalotin + Cefazolin": "other beta-lactam",
    "Cefazolin": "other beta-lactam",
    "Cefepime": "other beta-lactam",
    "Cefixime": "other beta-lactam",
    "Cefotaxime": "other beta-lactam",
    "Cefoxitin": "other beta-lactam",
    "Cefpodoxime": "other beta-lactam",
    "Ceftaroline fosamil": "other beta-lactam",
    "Ceftazidime": "other beta-lactam",
    "Ceftazidime + Avibactam": "other beta-lactam",
    "Ceftobiprole": "other beta-lactam",
    "Ceftolozane + Tazobactam": "other beta-lactam",
    "Ceftriaxone": "other beta-lactam",
    "Cefuroxime": "other beta-lactam",
    "Chloramphenicol": "other",
    "Ciprofloxacin": "quinolone",
    "Clarithromycin": "macrolides",
    "Clindamycin": "macrolides",
    "Colistin": "other",
    "Cotrimoxazole": "other",
    "Daptomycin": "other",
    "Doxycycline": "tetracyclin",
    "Ertapenem": "other beta-lactam",
    "Erythromycin": "macrolides",
    "Fluconazole": "antimycotic systemic",
    "Fosfomycin": "other",
    "Fosfomycin Tromethamine": "other",
    "Fusidic acid": "other",
    "Gentamicin": "aminoglycoside",
    "Imipenem": "other beta-lactam",
    "Isavuconazole": "antimycotic systemic",
    "Itraconazole": "antimycotic systemic",
    "Levofloxacin": "quinolone",
    "Linezolid": "other",
    "Meropenem": "other beta-lactam",
    "Meropenem + Vaborbactam": "other beta-lactam",
    "Metronidazole": "other",
    "Micafungin": "antimycotic systemic",
    "Minocycline": "tetracyclin",
    "Moxifloxacin": "quinolone",
    "Mupirocin": "other",
    "Nitrofurantoin": "other",
    "Norfloxacin": "quinolone",
    "Novobiocin": "other",
    "Oxacillin": "penicillins",
    "Penicillin": "penicillins",
    "Piperacillin": "penicillins",
    "Piperacillin-Tazobactam": "penicillins",
    "Polymyxin B": "other",
    "Posaconazole": "antimycotic systemic",
    "Rifampicin": "other",
    "Streptomycin": "other",
    "Teicoplanin": "other",
    "Telithromycin": "macrolides",
    "Tetracycline": "tetracyclin",
    "Ticarcillin": "penicillins",
    "Ticarcillin-clavulanic Acid": "penicillins",
    "Tigecycline": "tetracyclin",
    "Tobramycin": "aminoglycoside",
    "Vancomycin": "other",
    "Voriconazole": "antimycotic systemic",
}


# Originally obtained via pubchempy
drug_to_smiles = {
    "Amoxicillin + Clavulanic acid": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O.O=C(O)C1C(=CCO)OC2CC(=O)N21",
    "Ampicillin": "CC1(C)SC2C(NC(=O)C(N)c3ccccc3)C(=O)N2C1C(=O)O",
    "Ceftazidime": "CC(C)(ON=C(C(=O)NC1C(=O)N2C(C(=O)[O-])=C(C[n+]3ccccc3)CSC12)c1csc(N)n1)C(=O)O",
    "Ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Gentamicin": "CNC(C)C1CCC(N)C(OC2C(N)CC(N)C(OC3OCC(C)(O)C(NC)C3O)C2O)O1",
    "Ceftriaxone": "CON=C(C(=O)NC1C(=O)N2C(C(=O)O)=C(CSc3nc(=O)c(=O)[nH]n3C)CSC12)c1csc(N)n1",
    "Cefuroxime": "CON=C(C(=O)NC1C(=O)N2C(C(=O)O)=C(COC(N)=O)CSC12)c1ccco1",
    "Nitrofurantoin": "O=C1CN(N=Cc2ccc([N+](=O)[O-])o2)C(=O)N1",
    "Fosfomycin": "CC1OC1P(=O)(O)O",
    "Norfloxacin": "CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)cc21",
    "Polymyxin B": "CCC(C)CCCCC(=O)NC(CCN)C(=O)NC(C(=O)NC(CCN)C(=O)NC1CCNC(=O)C(C(C)O)NC(=O)C(CCN)NC(=O)C(CCN)NC(=O)C(CC(C)C)NC(=O)C(Cc2ccccc2)NC(=O)C(CCN)NC1=O)C(C)O",
    "Cotrimoxazole": "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC.Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
    "Clarithromycin": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(OC)CC(C)C(=O)C(C)C(O)C1(C)O",
    "Clindamycin": "CCCC1CC(C(=O)NC(C(C)Cl)C2OC(SC)C(O)C(O)C2O)N(C)C1",
    "Doxycycline": "CC1c2cccc(O)c2C(O)=C2C(=O)C3(O)C(O)=C(C(N)=O)C(=O)C(N(C)C)C3C(O)C21",
    "Fusidic acid": "CC(=O)OC1CC2(C)C(CC(O)C3C4(C)CCC(O)C(C)C4CCC32C)C1=C(CCC=C(C)C)C(=O)O",
    "Oxacillin": "Cc1onc(-c2ccccc2)c1C(=O)NC1C(=O)N2C1SC(C)(C)C2C(=O)O",
    "Penicillin": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "Rifampicin": "COC1C=COC2(C)Oc3c(C)c(O)c4c(O)c(c(C=NN5CCN(C)CC5)c(O)c4c3C2=O)NC(=O)C(C)=CC=CC(C)C(O)C(C)C(O)C(C)C(OC(C)=O)C1C",
    "Vancomycin": "CNC(CC(C)C)C(=O)NC1C(=O)NC(CC(N)=O)C(=O)NC2C(=O)NC3C(=O)NC(C(=O)NC(C(=O)O)c4cc(O)cc(O)c4-c4cc3ccc4O)C(O)c3ccc(c(Cl)c3)Oc3cc2cc(c3OC2OC(CO)C(O)C(O)C2OC2CC(C)(N)C(O)C(C)O2)Oc2ccc(cc2Cl)C1O",
    "Linezolid": "CC(=O)NCC1CN(c2ccc(N3CCOCC3)c(F)c2)C(=O)O1",
    "Mupirocin": "CC(=CC(=O)OCCCCCCCCC(=O)O)CC1OCC(CC2OC2C(C)C(C)O)C(O)C1O",
    "Piperacillin-Tazobactam": "CC1(Cn2ccnn2)C(C(=O)O)N2C(=O)CC2S1(=O)=O.CCN1CCN(C(=O)NC(C(=O)NC2C(=O)N3C2SC(C)(C)C3C(=O)[O-])c2ccccc2)C(=O)C1=O.[Na+]",
    "Metronidazole": "Cc1ncc([N+](=O)[O-])n1CCO",
    "Moxifloxacin": "COc1c(N2CC3CCCNC3C2)c(F)cc2c(=O)c(C(=O)O)cn(C3CC3)c12",
    "Amikacin": "NCCC(O)C(=O)NC1CC(N)C(OC2OC(CN)C(O)C(O)C2O)C(O)C1OC1OC(CO)C(O)C(N)C1O",
    "Cefepime": "CON=C(C(=O)NC1C(=O)N2C(C(=O)[O-])=C(C[N+]3(C)CCCC3)CSC12)c1csc(N)n1",
    "Imipenem": "CC(O)C1C(=O)N2C(C(=O)O)=C(SCCN=CN)CC12",
    "Azithromycin": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)CN(C)C(C)C(O)C1(C)O",
    "Erythromycin": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O",
    "Cefalotin + Cefazolin": "CC(=O)OCC1=C(C(=O)O)N2C(=O)C(NC(=O)Cc3cccs3)C2SC1.Cc1nnc(SCC2=C(C(=O)O)N3C(=O)C(NC(=O)Cn4cnnn4)C3SC2)s1",
    "Tetracycline": "CN(C)C1C(=O)C(C(N)=O)=C(O)C2(O)C(=O)C3=C(O)c4c(O)cccc4C(C)(O)C3CC12",
    "Novobiocin": "COC1C(OC(N)=O)C(O)C(Oc2ccc3c(O)c(NC(=O)c4ccc(O)c(CC=C(C)C)c4)c(=O)oc3c2C)OC1(C)C",
    "Meropenem": "CC(O)C1C(=O)N2C(C(=O)O)=C(SC3CNC(C(=O)N(C)C)C3)C(C)C12",
    "Daptomycin": "CCCCCCCCCC(=O)NC(Cc1c[nH]c2ccccc12)C(=O)NC(CC(N)=O)C(=O)NC(CC(=O)O)C(=O)NC1C(=O)NCC(=O)NC(CCCN)C(=O)NC(CC(=O)O)C(=O)NC(C)C(=O)NC(CC(=O)O)C(=O)NCC(=O)NC(CO)C(=O)NC(C(C)CC(=O)O)C(=O)NC(CC(=O)c2ccccc2N)C(=O)OC1C",
    "5-Fluorocytosine": "Nc1[nH]c(=O)ncc1F",
    "Amphotericin B": "CC1C=CC=CC=CC=CC=CC=CC=CC(OC2OC(C)C(O)C(N)C2O)CC2OC(O)(CC(O)CC(O)C(O)CCC(O)CC(O)CC(=O)OC(C)C(C)C1O)CC(O)C2C(=O)O",
    "Amoxicillin": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "Aztreonam": "CC1C(NC(=O)C(=NOC(C)(C)C(=O)O)c2csc(N)n2)C(=O)N1S(=O)(=O)O",
    "Caspofungin": "CCC(C)CC(C)CCCCCCCCC(=O)NC1CC(O)C(NCCN)NC(=O)C2C(O)CCN2C(=O)C(C(O)CCN)NC(=O)C(C(O)C(O)c2ccc(O)cc2)NC(=O)C2CC(O)CN2C(=O)C(C(C)O)NC1=O",
    "Ceftolozane + Tazobactam": "CC1(Cn2ccnn2)C(C(=O)O)N2C(=O)CC2S1(=O)=O.Cn1c(N)c(NC(=O)NCCN)c[n+]1CC1=C(C(=O)[O-])N2C(=O)C(NC(=O)C(=NOC(C)(C)C(=O)O)c3nsc(N)n3)C2SC1",
    "Colistin": "CCC(C)CCCC(=O)NC(CCN)C(=O)NC(C(=O)NC(CCN)C(=O)NC1CCNC(=O)C(C(C)O)NC(=O)C(CCN)NC(=O)C(CCN)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(CCN)NC1=O)C(C)O",
    "Cefotaxime": "CON=C(C(=O)NC1C(=O)N2C(C(=O)O)=C(COC(C)=O)CSC12)c1csc(N)n1",
    "Ertapenem": "CC(O)C1C(=O)N2C(C(=O)O)=C(SC3CNC(C(=O)Nc4cccc(C(=O)O)c4)C3)C(C)C12",
    "Fluconazole": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
    "Cefoxitin": "COC1(NC(=O)Cc2cccs2)C(=O)N2C(C(=O)O)=C(COC(N)=O)CSC21",
    "Levofloxacin": "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    "Minocycline": "CN(C)c1ccc(O)c2c1CC1CC3C(N(C)C)C(=O)C(C(N)=O)=C(O)C3(O)C(=O)C1=C2O",
    "Teicoplanin": "CCCCCCCCCC(=O)NC1C(Oc2c3cc4cc2Oc2ccc(cc2Cl)C(OC2OC(CO)C(O)C(O)C2NC(C)=O)C2NC(=O)C(NC(=O)C4NC(=O)C4NC(=O)C(Cc5ccc(c(Cl)c5)O3)NC(=O)C(N)c3ccc(O)c(c3)Oc3cc(O)cc4c3)c3ccc(O)c(c3)-c3c(OC4OC(CO)C(O)C(O)C4O)cc(O)cc3C(C(=O)O)NC2=O)OC(CO)C(O)C1O",
    "Tigecycline": "CN(C)c1cc(NC(=O)CNC(C)(C)C)c(O)c2c1CC1CC3C(N(C)C)C(=O)C(C(N)=O)=C(O)C3(O)C(=O)C1=C2O",
    "Tobramycin": "NCC1OC(OC2C(N)CC(N)C(OC3OC(CO)C(O)C(N)C3O)C2O)C(N)CC1O",
    "Voriconazole": "CC(c1ncncc1F)C(O)(Cn1cncn1)c1ccc(F)cc1F",
    "Ceftazidime + Avibactam": "CC(C)(ON=C(C(=O)NC1C(=O)N2C(C(=O)[O-])=C(C[n+]3ccccc3)CSC12)c1csc(N)n1)C(=O)O.NC(=O)C1CCC2CN1C(=O)N2OS(=O)(=O)O",
    "Meropenem + Vaborbactam": "CC(O)C1C(=O)N2C(C(=O)O)=C(SC3CNC(C(=O)N(C)C)C3)C(C)C12.O=C(O)CC1CCC(NC(=O)Cc2cccs2)B(O)O1",
    "Chloramphenicol": "O=C(NC(CO)C(O)c1ccc([N+](=O)[O-])cc1)C(Cl)Cl",
    "Cefpodoxime": "COCC1=C(C(=O)O)N2C(=O)C(NC(=O)C(=NOC)c3csc(N)n3)C2SC1",
    "Piperacillin": "CCN1CCN(C(=O)NC(C(=O)NC2C(=O)N3C2SC(C)(C)C3C(=O)O)c2ccccc2)C(=O)C1=O",
    "Ampicillin-Sulbactam": "CC1(C)C(C(=O)O)C2C(=O)CC2S1(=O)=O.CC1(C)SC2C(NC(=O)C(N)c3ccccc3)C(=O)N2C1C(=O)O",
    "Ticarcillin-clavulanic Acid": "CC1(C)SC2C(NC(=O)C(C(=O)O)c3ccsc3)C(=O)N2C1C(=O)O.O=C(O)C1C(=CCO)OC2CC(=O)N21",
    "Telithromycin": "CCC1OC(=O)C(C)C(=O)C(C)C(OC2OC(C)CC(N(C)C)C2O)C(C)(OC)CC(C)C(=O)C(C)C2N(CCCCn3cnc(-c4cccnc4)c3)C(=O)OC12C",
    "Ticarcillin": "CC1(C)SC2C(NC(=O)C(C(=O)O)c3ccsc3)C(=O)N2C1C(=O)O",
    "Streptomycin": "CNC1C(OC2C(OC3C(O)C(O)C(N=C(N)N)C(O)C3N=C(N)N)OC(C)C2(O)C=O)OC(CO)C(O)C1O",
    "Posaconazole": "CCC(C(C)O)n1ncn(-c2ccc(N3CCN(c4ccc(OCC5COC(Cn6cncn6)(c6ccc(F)cc6F)C5)cc4)CC3)cc2)c1=O",
    "Itraconazole": "CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OCC5COC(Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O",
    "Anidulafungin": "CCCCCOc1ccc(-c2ccc(-c3ccc(C(=O)NC4CC(O)C(O)NC(=O)C5C(O)C(C)CN5C(=O)C(C(C)O)NC(=O)C(C(O)C(O)c5ccc(O)cc5)NC(=O)C5CC(O)CN5C(=O)C(C(C)O)NC4=O)cc3)cc2)cc1",
    "Micafungin": "CCCCCOc1ccc(-c2cc(-c3ccc(C(=O)NC4CC(O)C(O)NC(=O)C5C(O)C(C)CN5C(=O)C(C(O)CC(N)=O)NC(=O)C(C(O)C(O)c5ccc(O)c(OS(=O)(=O)O)c5)NC(=O)C5CC(O)CN5C(=O)C(C(C)O)NC4=O)cc3)no2)cc1",
    "Ampicillin + Amoxicillin": "CC1(C)SC2C(NC(=O)C(N)c3ccccc3)C(=O)N2C1C(=O)O.CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "Fosfomycin Tromethamine": "CC1OC1P(=O)(O)O.NC(CO)(CO)CO",
    "Cefazolin": "Cc1nnc(SCC2=C(C(=O)O)N3C(=O)C(NC(=O)Cn4cnnn4)C3SC2)s1",
    "Cefixime": "C=CC1=C(C(=O)O)N2C(=O)C(NC(=O)C(=NOCC(=O)O)c3csc(N)n3)C2SC1",
    "Ceftaroline fosamil": "CCON=C(C(=O)NC1C(=O)N2C(C(=O)[O-])=C(Sc3nc(-c4cc[n+](C)cc4)cs3)CSC12)c1nsc(NP(=O)(O)O)n1",
    "Ceftobiprole": "Nc1nc(C(=NO)C(=O)NC2C(=O)N3C(C(=O)O)=C(C=C4CCN(C5CCNC5)C4=O)CSC23)ns1",
    "Bacitracin A": "CCC(C)C(N)C1=NC(C(=O)NC(CC(C)C)C(=O)NC(CCC(=O)O)C(=O)NC(C(=O)NC2CCCCNC(=O)C(CC(N)=O)NC(=O)C(CC(=O)O)NC(=O)C(Cc3cnc[nH]3)NC(=O)C(Cc3ccccc3)NC(=O)C(C(C)CC)NC(=O)C(CCCN)NC2=O)C(C)CC)CS1",
    "Isavuconazole": "CC(c1nc(-c2ccc(C#N)cc2)cs1)C(O)(Cn1cncn1)c1cc(F)ccc1F",
}


def DRIAMS_raw_spectra_to_h5torch(DRIAMS_ROOT, outfile):
    print("(1) gathering all spectra files ...")
    ids = []
    for ix, (root, dirs, files_) in enumerate(os.walk(DRIAMS_ROOT)):
        if "raw" in root:
            for f in files_:  # walk through all raw files
                if f.endswith(".txt"):
                    k = root + "/" + f
                    ids.append("/".join(k.split("/")[-4:]))

    print("(2) gathering species data for all spectra files ...")
    alldata = []
    for root, dirs, files_ in os.walk(DRIAMS_ROOT):
        if "id" in root:
            for f in files_:
                if f.endswith("_clean.csv"):
                    print("... reading amr data for " + root + "/" + f)
                    k = root + "/" + f
                    data = pd.read_csv(k)

                    data["code"] = (
                        k.split("/")[-4]
                        + "/raw/"
                        + k.split("/")[-2]
                        + "/"
                        + data["code"]
                        + ".txt"
                    )
                    ids_driams_year = [
                        f
                        for f in ids
                        if (f.startswith(k.split("/")[-4]))
                        and (k.split("/")[-2] == f.split("/")[2])
                    ]
                    # only keep amr data for files that exist
                    keepers = [d in ids_driams_year for d in data["code"]]
                    alldata.append(data.iloc[keepers, :])

    # cleanup data
    alldata = pd.concat(alldata).drop(
        ["laboratory_species", "Unnamed: 0.1", "Unnamed: 0", "combined_code", "genus"],
        axis=1,
    )[["code", "species"]]

    # putting unidentified species to nans
    alldata.loc[
        alldata["species"].str.contains("not reliable identification"), ["species"]
    ] = np.nan

    alldata = pd.concat(
        [
            pd.DataFrame({"code": list(set(ids).difference(set(alldata["code"])))}),
            alldata,
        ]
    )
    dataraw = alldata

    species_labels = np.unique(dataraw["species"].astype(bytes))
    _, species = np.where(
        dataraw["species"].astype(bytes).values.reshape(-1, 1) == species_labels
    )

    loc = dataraw["code"].values.astype(bytes)

    data_path = files("maldi_nn.utils").joinpath("driams_split.json")
    split = json.load(open(data_path))

    spectrum_split = np.array(
        [
            split[l]
            for l in pd.Series(loc.astype(str))
            .str.split("/", expand=True)[[0, 2, 3]]
            .apply("/".join, axis=1)
            .values
        ]
    )

    f = h5torch.File(outfile, "w")

    f.register(species, "central")
    f.register(loc, 0, name="loc")
    f.register(species_labels, "unstructured", name="species_labels")
    f.register(spectrum_split.astype(bytes), "unstructured", name="split")

    ints = []
    mzs = []
    for ix, k in enumerate(loc):
        spectrum = pd.read_table(
            os.path.join(DRIAMS_ROOT, k.astype(str)),
            comment="#",
            sep=" ",
            index_col=None,
            header=0,
        ).values
        spectrum = spectrum[~np.isnan(spectrum).any(1)]

        mz = spectrum[:, 0]
        intensities = spectrum[:, 1]
        ints.append(intensities.astype(np.uint32))
        mzs.append(mz.astype(np.float32))
        if (ix + 1) % 1000 == 0:
            print(ix, end=" ", flush=True)
            if (ix + 1) == 1000:
                f.register(ints, 0, name="intensity", mode="vlen", length=len(species))
                f.register(mzs, 0, name="mz", mode="vlen", length=len(species))
                ints = []
                mzs = []
            else:
                f.append(ints, "0/intensity")
                f.append(mzs, "0/mz")
                ints = []
                mzs = []

    f.append(ints, "0/intensity")
    f.append(mzs, "0/mz")
    f.close()
    print("done")
    return None


def DRIAMS_raw_amr_to_h5torch(DRIAMS_ROOT, outfile):
    print("(1) gathering all spectra files ...")
    ids = []
    for ix, (root, dirs, files_) in enumerate(os.walk(DRIAMS_ROOT)):
        if "raw" in root:
            for f in files_:  # walk through all raw files
                if f.endswith(".txt"):
                    k = root + "/" + f
                    ids.append("/".join(k.split("/")[-4:]))

    print("(2) gathering amr data for all spectra files ...")
    alldata = []
    for root, dirs, files_ in os.walk(DRIAMS_ROOT):
        if "id" in root:
            for f in files_:
                if f.endswith("_clean.csv"):
                    print("... reading amr data for " + root + "/" + f)
                    k = root + "/" + f
                    data = pd.read_csv(k)

                    data["code"] = (
                        k.split("/")[-4]
                        + "/raw/"
                        + k.split("/")[-2]
                        + "/"
                        + data["code"]
                        + ".txt"
                    )
                    ids_driams_year = [
                        f
                        for f in ids
                        if (f.startswith(k.split("/")[-4]))
                        and (k.split("/")[-2] == f.split("/")[2])
                    ]
                    # only keep amr data for files that exist
                    keepers = [d in ids_driams_year for d in data["code"]]
                    alldata.append(data.iloc[keepers, :])

    # cleanup data
    alldata_ = pd.concat(alldata).drop(
        ["laboratory_species", "Unnamed: 0.1", "Unnamed: 0", "combined_code", "genus"],
        axis=1,
    )

    # delete all labels that are not I, R or S
    alldata_without = alldata_.drop(columns=["code", "species"])
    alldata_without[
        ~(
            (alldata_without == "R")
            | (alldata_without == "I")
            | (alldata_without == "S")
        )
    ] = np.nan
    alldata = pd.concat([alldata_[["code", "species"]], alldata_without], axis=1)
    alldata = alldata.loc[
        :, alldata.nunique(dropna=False) != 1
    ]  # delete columns without labels

    # renaming and merging columns that have identical chemical structures
    def merge(data, names):
        data[names[0]] = (
            data[names]
            .astype(str)
            .apply("_".join, axis=1)
            .map(
                {
                    "R_nan": "R",
                    "nan_R": "R",
                    "S_nan": "S",
                    "nan_S": "S",
                    "I_nan": "I",
                    "nan_I": "I",
                    "R_R": "R",
                    "S_S": "S",
                    "I_I": "I",
                }
            )
        )
        data = data.drop([names[1]], axis=1)
        return data

    alldata = alldata.drop(["Quinolones", "Aminoglycosides"], axis=1)

    alldata = merge(alldata, ["Levofloxacin", "Ofloxacin"])  # are stereoisomers

    alldata = merge(alldata, ["Benzylpenicillin", "Benzylpenicillin_others"])
    alldata = merge(alldata, ["Benzylpenicillin", "Benzylpenicillin_with_meningitis"])
    alldata = merge(alldata, ["Benzylpenicillin", "Benzylpenicillin_with_pneumonia"])

    alldata = merge(alldata, ["Penicillin", "Penicillin_with_endokarditis"])
    alldata = merge(alldata, ["Penicillin", "Penicillin_without_meningitis"])
    alldata = merge(alldata, ["Penicillin", "Penicillin_without_endokarditis"])
    alldata = merge(alldata, ["Penicillin", "Penicillin_with_pneumonia"])
    alldata = merge(alldata, ["Penicillin", "Penicillin_with_meningitis"])
    alldata = merge(alldata, ["Penicillin", "Penicillin_with_other_infections"])

    alldata = merge(alldata, ["Penicillin", "Benzylpenicillin"])

    alldata = merge(alldata, ["Cefuroxime", "Cefuroxime.1"])
    alldata = merge(alldata, ["Cotrimoxazole", "Cotrimoxazol"])
    alldata = merge(alldata, ["Gentamicin", "Gentamicin_high_level"])
    alldata = merge(alldata, ["Cefoxitin", "Cefoxitin_screen"])
    alldata = merge(alldata, ["Teicoplanin", "Teicoplanin_GRD"])
    alldata = merge(alldata, ["Vancomycin", "Vancomycin_GRD"])
    alldata = merge(alldata, ["Rifampicin", "Rifampicin_1mg-l"])

    alldata = merge(alldata, ["Meropenem", "Meropenem_with_meningitis"])
    alldata = merge(alldata, ["Meropenem", "Meropenem_without_meningitis"])
    alldata = merge(alldata, ["Meropenem", "Meropenem_with_pneumonia"])

    alldata = merge(
        alldata,
        [
            "Amoxicillin-Clavulanic acid",
            "Amoxicillin-Clavulanic acid_uncomplicated_HWI",
        ],
    )

    # renaming scheme
    alldata.rename(
        columns={
            "Strepomycin_high_level": "Streptomycin",
            "Bacitracin": "Bacitracin A",
            "Ceftarolin": "Ceftaroline fosamil",
            "Fosfomycin-Trometamol": "Fosfomycin Tromethamine",
            "Amoxicillin-Clavulanic acid": "Amoxicillin + Clavulanic acid",
            "Ceftolozane-Tazobactam": "Ceftolozane + Tazobactam",
            "Ceftazidime-Avibactam": "Ceftazidime + Avibactam",
            "Meropenem-Vaborbactam": "Meropenem + Vaborbactam",
            "Ticarcillin-Clavulan acid": "Ticarcillin-clavulanic Acid",
            "Ampicillin-Amoxicillin": "Ampicillin + Amoxicillin",
            "Cefalotin-Cefazolin": "Cefalotin + Cefazolin",
        },
        inplace=True,
    )

    # putting unidentified species to nans
    alldata.loc[
        alldata["species"].str.contains("not reliable identification"), ["species"]
    ] = np.nan
    alldata = pd.concat(
        [
            pd.DataFrame({"code": list(set(ids).difference(set(alldata["code"])))}),
            alldata,
        ]
    )
    dataraw = alldata

    dataraw = dataraw[(~pd.isnull(dataraw.iloc[:, 2:])).sum(1) != 0]

    species_labels = np.unique(dataraw["species"].astype(bytes))
    _, species = np.where(
        dataraw["species"].astype(bytes).values.reshape(-1, 1) == species_labels
    )

    loc = dataraw["code"].values.astype(bytes)
    amr_row, amr_col = np.where(~pd.isnull(dataraw.iloc[:, 2:]))
    amr_val = dataraw.iloc[:, 2:].values[amr_row, amr_col].astype(bytes)
    amr_names = dataraw.columns[2:].values.astype(bytes)
    amr_classes = [drug_to_class[k.astype(str)] for k in amr_names]
    amr_smiles = [drug_to_smiles[k.astype(str)] for k in amr_names]

    data_path = files("maldi_nn.utils").joinpath("driams_split.json")
    split = json.load(open(data_path))

    spectrum_split = np.array(
        [
            split[l]
            for l in pd.Series(loc.astype(str))
            .str.split("/", expand=True)[[0, 2, 3]]
            .apply("/".join, axis=1)
            .values
        ]
    )

    amr_split = spectrum_split[amr_row]
    indices = np.stack([amr_row, amr_col])
    shp = list(indices.max(1) + 1)

    f = h5torch.File(outfile, "w")

    f.register((indices, amr_val, shp), "central", mode="coo")
    f.register(species, 0, name="species")
    f.register(loc, 0, name="loc")
    f.register(species_labels, "unstructured", name="species_labels")
    f.register(np.array(amr_smiles).astype(bytes), 1, name="drug_smiles")
    f.register(np.array(amr_classes).astype(bytes), 1, name="drug_classes")
    f.register(amr_names.astype(bytes), 1, name="drug_names")
    f.register(amr_split.astype(bytes), "unstructured", name="split")

    ints = []
    mzs = []
    for ix, k in enumerate(loc):
        spectrum = pd.read_table(
            os.path.join(DRIAMS_ROOT, k.astype(str)),
            comment="#",
            sep=" ",
            index_col=None,
            header=0,
        ).values
        spectrum = spectrum[~np.isnan(spectrum).any(1)]

        mz = spectrum[:, 0]
        intensities = spectrum[:, 1]
        ints.append(intensities.astype(np.uint32))
        mzs.append(mz.astype(np.float32))
        if (ix + 1) % 1000 == 0:
            print(ix, end=" ", flush=True)
            if (ix + 1) == 1000:
                f.register(ints, 0, name="intensity", mode="vlen", length=len(species))
                f.register(mzs, 0, name="mz", mode="vlen", length=len(species))
                ints = []
                mzs = []
            else:
                f.append(ints, "0/intensity")
                f.append(mzs, "0/mz")
                ints = []
                mzs = []

    f.append(ints, "0/intensity")
    f.append(mzs, "0/mz")
    f.close()

    print("done")
    return None


def DRIAMS_raw_to_binned(rawfile, processed_file):
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


def DRIAMS_raw_to_peaks(rawfile, processed_file):
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
    DRIAMS_root = str(sys.argv[1])
    amrraw = str(sys.argv[2])
    spectraraw = str(sys.argv[3])
    amrbin = str(sys.argv[4])
    spectrabin = str(sys.argv[5])
    amrpks = str(sys.argv[6])
    spectrapks = str(sys.argv[7])

    DRIAMS_raw_amr_to_h5torch(DRIAMS_root, amrraw)
    DRIAMS_raw_spectra_to_h5torch(DRIAMS_root, spectraraw)

    DRIAMS_raw_to_binned(amrraw, amrbin)
    DRIAMS_raw_to_peaks(amrraw, amrpks)
    DRIAMS_raw_to_binned(spectraraw, spectrabin)
    DRIAMS_raw_to_peaks(spectraraw, spectrapks)


if __name__ == "__main__":
    main()
