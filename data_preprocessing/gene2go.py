import pandas as pd
from collections import defaultdict
import re

gaf_file = "./data/GO_data/goa_human.gaf"
df = pd.read_csv(gaf_file, sep="\t", comment="!", header=None, dtype=str)
df.columns = [
    "DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
    "DB_Reference", "Evidence_Code", "With_or_From", "Aspect",
    "DB_Object_Name", "Synonym", "DB_Object_Type", "Taxon",
    "Date", "Assigned_By", "Annotation_Extension",
    "Gene_Product_Form_ID"
]

df = df[df["DB"] == "UniProtKB"]

df = df[~df["DB_Object_ID"].str.startswith("URS")]

df = df[df["DB_Object_Symbol"].notna()]
df = df[df["DB_Object_Symbol"].str.strip() != ""]

gene_pattern = re.compile(r"^[A-Za-z0-9\-]+$")

df = df[df["DB_Object_Symbol"].apply(lambda x: bool(gene_pattern.match(x)))]

gene2go = defaultdict(set)

for _, row in df.iterrows():
    gene_symbol = row["DB_Object_Symbol"]
    go_id = row["GO_ID"]
    gene2go[gene_symbol].add(go_id)

print("Total human genes (filtered):", len(gene2go))

for i, (gene, gos) in enumerate(gene2go.items()):
    print(gene, list(gos)[:5])
    if i >= 5:
        break

gene_go_df = pd.DataFrame([
    {"Gene": gene, "GO_terms": ";".join(sorted(list(gos)))}
    for gene, gos in gene2go.items()
])

gene_go_df.to_csv("../data/GO_data/gene2go.csv", index=False)
print("Saved to gene2go.csv")
