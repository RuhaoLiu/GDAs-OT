import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from goatools.base import get_godag
import numpy as np
import pickle
from tqdm import tqdm
import re


def clean_definition(defn):
    return re.sub(r"\[[^\]]+\]$", "", defn).strip()


def embed_texts(texts, batch_size=64):
    all_embeds = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        all_embeds.append(batch_emb)

    return np.vstack(all_embeds)


def save_common_feature(emb_matrix, gene_name):

    graph_node_name = np.load("../data/HumanNet_data/gene_nodelist.npy")
    go_node2idx = {gene: i for i, gene in enumerate(gene_name)}
    common_genes = [gene for gene in graph_node_name if gene in go_node2idx]
    np.save('../data/GO_data/common_genes.npy', common_genes)
    aligned_features = np.array([emb_matrix[go_node2idx[gene]] for gene in common_genes])
    df = pd.DataFrame(aligned_features)
    df.insert(0, "gene", common_genes)  # add gene name
    print('save gene2embedding file...')
    df.to_csv("../data/GO_data/gene_feature.csv", index=False)


def gene2embeddings(gene2go_path):

    gene2go = pd.read_csv(gene2go_path)

    godag = get_godag(
        "../data/GO_data/go.obo",
        optional_attrs={'defn'}
    )

    # GO_ID → Definition
    go2def = {
        go.id: go.defn
        for go in godag.values()
        if hasattr(go, "defn") and go.defn is not None
    }

    print('start gene2embedding progressing...')

    go_ids = list(go2def.keys())
    go_texts = [clean_definition(go2def[g]) for g in go_ids]
    go_emb_matrix = embed_texts(go_texts)
    go_embeddings = {go_ids[i]: go_emb_matrix[i] for i in range(len(go_ids))}

    gene_embeddings = {}
    for _, row in gene2go.iterrows():
        gene = row['Gene']
        go_terms = row['GO_terms'].split(';')
        valid_terms = [t for t in go_terms if t in go_embeddings]
        if not valid_terms:
            continue

        # embedding
        embeds = np.stack([go_embeddings[t] for t in valid_terms])
        gene_embeddings[gene] = embeds.mean(axis=0)

    genes = list(gene_embeddings.keys())
    emb_matrix = np.stack([gene_embeddings[g] for g in genes])
    save_common_feature(emb_matrix, genes)


def disease2embeddings(do_graph_path):

    with open(do_graph_path, "rb") as f:
        do_graph = pickle.load(f)

    print('start dis2embedding progressing...')
    nodes = list(do_graph.nodes())
    definitions = [do_graph.nodes[n].get("definition", "") for n in nodes]
    embeds = embed_texts(definitions)

    print('save dis2embedding file...')
    np.save("../data/DO_data/disease_embeddings.npy", embeds)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load PudMedbert model
    tokenizer = AutoTokenizer.from_pretrained("../PubMedBert/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("../PubMedBert/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model.to(device)
    model.eval()

    # gene file
    gene2go_path = '../data/GO_data/gene2go.csv'

    # disease file
    do_graph_path = '../data/DO_data/do_graph.pkl'

    gene2embeddings(gene2go_path)
    disease2embeddings(do_graph_path)
    print("finished")
