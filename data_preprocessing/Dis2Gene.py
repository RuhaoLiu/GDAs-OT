import pronto
import pandas as pd
from tqdm import tqdm

#  read DO data
do_obo_file = "../data/DO_data/doid.obo"
ontology = pronto.Ontology(do_obo_file)

do_umls_to_doid = {}  # UMLS_CUI -> DOID
do_name_to_doid = {}  # term name / synonym -> DOID

for term in tqdm(ontology.terms(), desc="Extracting DOID info"):
    if term.obsolete:
        continue
    # extracting UMLS_CUI_id
    for xref in term.xrefs:
        if xref.id.startswith("UMLS_CUI:"):
            cui = xref.id.split(":")[1]
            do_umls_to_doid[cui] = term.id

    # disease name : DO_id
    do_name_to_doid[term.name.lower().strip()] = term.id
    for syn in term.synonyms:
        syn_name = syn.description.lower().strip()
        do_name_to_doid[syn_name] = term.id

# read disGeNet data
disgenet_file = "../data/all_gene_disease_associations.tsv"
disgenet_data = pd.read_csv(disgenet_file, sep="\t", dtype=str)

# UMLS_CUI_id matching
df_umls = disgenet_data[disgenet_data["diseaseId"].isin(do_umls_to_doid)].copy()
df_umls["DOID"] = df_umls["diseaseId"].map(do_umls_to_doid)

# processing no match data using disease name
df_missing = disgenet_data[~disgenet_data["diseaseId"].isin(do_umls_to_doid)].copy()
df_missing["diseaseName_norm"] = df_missing["diseaseName"].str.lower().str.strip()
df_name_matched = df_missing[df_missing["diseaseName_norm"].isin(do_name_to_doid)].copy()
df_name_matched["DOID"] = df_name_matched["diseaseName_norm"].map(do_name_to_doid)

# final data
final_df = pd.concat([df_umls, df_name_matched], ignore_index=True)
final_df.drop(columns=["diseaseType", "diseaseClass", "YearInitial", "YearFinal", "NofPmids",
                       "NofSnps", "source", "diseaseName_norm"]).to_csv('../data/Dis2Gene.tsv', sep='\t', index=False)

