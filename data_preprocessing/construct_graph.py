import os
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import pronto


def construct_do_graph(file_path):
    # read DO data
    ontology = pronto.Ontology(file_path)

    do_G = nx.DiGraph()

    for term in ontology.terms():
        if term.obsolete:
            continue
        node_id = term.id
        label = term.name
        definition = term.definition if term.definition else ""
        synonyms = [syn.description for syn in term.synonyms]
        subsets = [s for s in term.subsets]
        xrefs = [xref.id for xref in term.xrefs]

        do_umls_id = []
        for xr in xrefs:
            if xr.split(":")[0] == 'UMLS_CUI':
                do_umls_id.append(xr.split(":")[1])
        # add node
        do_G.add_node(node_id, label=label, definition=definition,
                      synonyms=synonyms, subsets=subsets, xrefs=xrefs)

        # add edge(parent relationships)
        for parent in term.superclasses(distance=1):
            if parent.id != node_id:
                do_G.add_edge(node_id, parent.id, type="is_a")
                # G.add_edge(parent.id, node_id, type="inverse_is_a")

    print("Node:", do_G.number_of_nodes())
    print("Edge:", do_G.number_of_edges())

    with open("../data/DO_data/do_graph.pkl", "wb") as f:
        pickle.dump(do_G, f)

    # to sparse adjacency matrix
    nodelist = list(do_G.nodes())
    do_A_coo = nx.adjacency_matrix(do_G, nodelist=nodelist).tocoo()

    np.save("../data/DO_data/disease_nodelist.npy", np.array(nodelist))
    with open("../data/DO_data/do_graph_sparse.pkl", "wb") as f:
        pickle.dump(do_A_coo, f)


def construct_gene_graph(gene_file):

    df = pd.read_csv(gene_file, sep="\t", names=["gene1", "gene2", "weight"])
    common_nodes = np.load('../data/GO_data/common_genes.npy')
    gene_G = nx.DiGraph()

    for _, row in df.iterrows():
        g1, g2 = row["gene1"], row["gene2"]
        if g1 in common_nodes and g2 in common_nodes:
            gene_G.add_edge(g1, g2, weight=1.0)

    nodelist = list(gene_G.nodes())
    # gene_A_coo = nx.adjacency_matrix(gene_G, nodelist=nodelist).tocoo()

    np.save("../data/HumanNet_data/gene_nodelist_common.npy", np.array(nodelist))
    with open("../data/HumanNet_data/humannet_graph_common.pkl", "wb") as f:
        pickle.dump(gene_G, f)


if __name__ == '__main__':

    # disease file path
    do_file = "../data/DO_data/doid.obo"

    # gene file path
    gene_file = "../data/HumanNet_data/HumanNet-XC.symbol.tsv"

    # construct disease and gene graph
    construct_do_graph(do_file)
    construct_gene_graph(gene_file)
