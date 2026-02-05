import pickle
import pandas as pd
import torch
import argparse
import numpy as np
from model.Dis2GeneLearning import GDALearning


def main(args):

    # load genes disease data
    gene_feat = pd.read_csv('data/GO_data/gene_feature.csv').iloc[:, 1:]
    dis_feat = np.load('data/DO_data/disease_embeddings.npy')
    gene_feat = torch.tensor(gene_feat.values, dtype=torch.float32)

    dis_feat = torch.tensor(dis_feat, dtype=torch.float32)
    dis2gene = pd.read_csv('data/Dis2Gene.tsv', sep="\t")[['geneSymbol', 'DOID']]
    # dis2gene = pd.read_csv('data/Dis2Gene_curated.tsv', sep="\t")[['geneSymbol', 'DOID']]

    with open('data/HumanNet_data/humannet_graph_common.pkl', 'rb') as f:
        gene_graph = pickle.load(f)

    with open('data/DO_data/do_graph.pkl', 'rb') as f:
        dis_graph = pickle.load(f)

    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    # hyperparameter
    arg_dict = {'dis_graph': dis_graph,
                'gene_graph': gene_graph,
                'dis_feat': dis_feat,
                'gene_feat': gene_feat,
                'device': device,
                'has_ot_matrix': args.has_ot_matrix,
                'dis2gene': dis2gene,
                'GCN_epochs': args.GCN_epochs,
                'MLP_epochs': args.MLP_epochs,
                'cost_type': args.cost_type,
                'loss_type': args.loss_type,
                'lr': args.lr,
                'alpha': args.alpha,
                'beta': args.beta,
                'margin': args.margin,
                'sample_num': args.sample_num,
                'outer_iter': args.outer_iter,
                'random_seed': 50
                }

    gda_model = GDALearning(arg_dict)
    if args.model_controller == 'gda_prediction':
        gda_model.gda_prediction()
    if args.model_controller == 'load_model':
        gda_model.load_model()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--GCN_epochs',
        help='number of GCN epochs',
        default=20,
        type=int)
    parser.add_argument(
        '--MLP_epochs',
        help='number of MLP epochs',
        default=15,
        type=int)
    parser.add_argument(
        '--cost_type',
        help='Euclidean distance or cosine',
        default='cosine',
        type=str)
    parser.add_argument(
        '--loss_type',
        help='L2 or KL',
        default='L2',
        type=str)
    parser.add_argument(
        '--lr',
        help='',
        default=1e-3,
        type=float)
    parser.add_argument(
        '--alpha',
        help='',
        default=0.1,
        type=float)
    parser.add_argument(
        '--beta',
        help='',
        default=5.0,
        type=float)
    parser.add_argument(
        '--margin',
        help='',
        default=3.0,
        type=float)
    parser.add_argument(
        '--sample_num',
        help='number of positive and negative sample',
        default=10000,
        type=int)
    parser.add_argument(
        '--outer_iter',
        help='GW maximum iter',
        default=50,
        type=int)
    parser.add_argument(
        '--has_ot_matrix',
        help='use precomputed OT matrix',
        action='store_true')
    parser.add_argument(
        '--model_controller',
        choices=['gda_prediction', 'load_model'],
        default='gda_prediction',
        type=str)

    args = parser.parse_args()
    main(args)

