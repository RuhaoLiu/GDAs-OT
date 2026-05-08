import os
import pickle
import random
import numpy as np
import torch
import copy
import matplotlib
from sklearn.decomposition import PCA
from tqdm import tqdm

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score, precision_score, \
    recall_score
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .gcn import GCNConv


class GDALearning(object):

    def __init__(self, arg_dic):
        self.dis_graph = arg_dic['dis_graph']
        self.gene_graph = arg_dic['gene_graph']
        self.dis_feat = arg_dic['dis_feat']
        self.gene_feat = arg_dic['gene_feat']
        self.device = arg_dic['device']
        self.has_ot_matrix = arg_dic['has_ot_matrix']
        self.dis2gene = arg_dic['dis2gene']
        self.GCN_epochs = arg_dic['GCN_epochs']
        self.MLP_epochs = arg_dic['MLP_epochs']
        self.cost_type = arg_dic['cost_type']
        self.loss_type = arg_dic['loss_type']
        self.lr = arg_dic['lr']
        self.alpha = arg_dic['alpha']
        self.beta = arg_dic['beta']
        self.margin = arg_dic['margin']
        self.sample_num = arg_dic['sample_num']
        self.outer_iter = arg_dic['outer_iter']
        self.random_seed = arg_dic['random_seed']
        self.accs = []
        self.aucs = []
        self.f1_scores = []
        self.auprs = []
        self.gcn_train_loss = []
        # self.gcn_test_loss = []
        self.mlp_train_loss = []
        # self.mlp_test_loss = []
        self.epoch_aucs = []

    def get_edge_embeddings(self, G, X, k=5, nodelist=None):

        if nodelist is None:
            nodelist = list(G.nodes())
        node2idx = {n: i for i, n in enumerate(nodelist)}
        N = len(nodelist)

        knn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        knn.fit(X)
        distances, neighbors = knn.kneighbors(X)

        pos_edges = set()
        for u, v in G.edges():
            ui, vi = node2idx[u], node2idx[v]
            a, b = sorted((ui, vi))
            pos_edges.add((a, b))

        unlabeled_edges = set()

        for i in range(N):
            for j in neighbors[i][1:]:
                a, b = sorted((i, j))
                if (a, b) in pos_edges:
                    continue
                else:
                    unlabeled_edges.add((a, b))

        all_edges = []
        for e in pos_edges:
            all_edges.append((*e, 1))
        for e in unlabeled_edges:
            all_edges.append((*e, 0))

        # build edge embeddings
        features = []
        labels = []
        edge_index = []
        for u, v, label in all_edges:
            features.append(np.concatenate([X[u], X[v]], axis=0))
            labels.append(label)
            edge_index.append((u, v))
        return np.array(features), np.array(labels), np.array(edge_index)

    def self_cost_mat(self, embs, cost_type, device):

        embs = embs.to(device)
        if cost_type == 'cosine':
            # cosine similarity
            energy = torch.sqrt(torch.sum(embs ** 2, dim=1, keepdim=True))  # (batch_size, 1)
            cost = 1 - torch.exp(
                -5 * (1 - torch.matmul(embs, torch.t(embs)) / (torch.matmul(energy, torch.t(energy)) + 1e-5)))
        else:
            # Euclidean distance
            embs = torch.matmul(embs, torch.t(embs))  # (batch_size, batch_size)
            embs_diag = torch.diag(embs).view(-1, 1).repeat(1, embs.size(0))  # (batch_size, batch_size)
            cost = 1 - torch.exp(-(embs_diag + torch.t(embs_diag) - 2 * embs) / embs.size(1))
        return cost

    def mutual_cost_mat(self, embs1, embs2, cost_type, device):

        embs1 = embs1.to(device)
        embs2 = embs2.to(device)
        if cost_type == 'cosine':
            # cosine similarity
            energy1 = torch.sqrt(torch.sum(embs1 ** 2, dim=1, keepdim=True))  # (batch_size1, 1)
            energy2 = torch.sqrt(torch.sum(embs2 ** 2, dim=1, keepdim=True))  # (batch_size2, 1)
            cost = 1 - torch.exp(
                -(1 - torch.matmul(embs1, torch.t(embs2)) / (torch.matmul(energy1, torch.t(energy2)) + 1e-5)))
        else:
            # Euclidean distance
            embs = torch.matmul(embs1, torch.t(embs2))  # (batch_size1, batch_size2)
            # (batch_size1, batch_size2)
            embs_diag1 = torch.diag(torch.matmul(embs1, torch.t(embs1))).view(-1, 1).repeat(1, embs2.size(0))
            # (batch_size2, batch_size1)
            embs_diag2 = torch.diag(torch.matmul(embs2, torch.t(embs2))).view(-1, 1).repeat(1, embs1.size(0))
            cost = 1 - torch.exp(-(embs_diag1 + torch.t(embs_diag2) - 2 * embs) / embs1.size(1))
        return cost

    def pairs_to_index(self, pairs, gene_node2idx, dis_node2idx):

        idx_pairs = []

        for g, d in pairs:
            if g in gene_node2idx and d in dis_node2idx:
                idx_pairs.append((gene_node2idx[g], dis_node2idx[d]))

        return idx_pairs

    def sinkhorn_iter(self, mu_s, mu_t, cost, display=True, reg=0.1, max_iter=300, tol=1e-9):

        K = torch.exp(-cost / reg)
        u = mu_s.sum().repeat(mu_s.size(0), 1)
        u /= u.sum()
        v = 0

        for it in range(max_iter):
            u_prev = u.clone()
            v = mu_t / torch.matmul(torch.t(K), u)
            u = mu_s / torch.matmul(K, v)

            if it % 100 == 0 and display:
                err = torch.norm(u - u_prev).item()
                print(f"Iter {it}, err = {err:.3e}")
                if err < tol:
                    break

        T = torch.matmul(torch.matmul(torch.diag(u[:, 0]), K), torch.diag(v[:, 0]))
        return T

    def wasserstein_distance(self, P, U, cost, device):

        p = len(P)
        u = len(U)
        print(f"positive number: {p}, unlabeled number: {u}")
        mu_s = torch.ones(p, device=device) / p  # source
        mu_s = mu_s.unsqueeze(1)
        mu_t = torch.ones(u, device=device) / u  # target
        mu_t = mu_t.unsqueeze(1)

        # Sinkhorn OT
        T = self.sinkhorn_iter(mu_s, mu_t, cost)

        return T

    def gromov_wasserstein_discrepancy(self, cost_s, cost_t, loss_type):

        mu_s = torch.ones(cost_s.size(0), dtype=torch.float32, device=self.device) / cost_s.size(0)
        mu_s = mu_s.unsqueeze(1)
        mu_t = torch.ones(cost_t.size(0), dtype=torch.float32, device=self.device) / cost_t.size(0)
        mu_t = mu_t.unsqueeze(1)
        ns = mu_s.size(0)
        nt = mu_t.size(0)
        T = torch.ones(ns, nt, dtype=torch.float32, device=self.device) / (ns * nt)

        if loss_type == 'L2':
            # f1(a) = a^2, f2(b) = b^2, h1(a) = a, h2(b) = 2b
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(cost_s ** 2, mu_s).repeat(1, nt)
            f2_st = torch.matmul(torch.t(mu_t), torch.t(cost_t ** 2)).repeat(ns, 1)
            cost_st = f1_st + f2_st
            if not self.has_ot_matrix:
                T_prev = T.clone()
                for t in tqdm(range(self.outer_iter), desc="Outer Iter"):
                    cost = cost_st - 2 * torch.matmul(torch.matmul(cost_s, T), torch.t(cost_t))
                    with torch.no_grad():
                        T = self.sinkhorn_iter(mu_s, mu_t, cost, display=False, max_iter=1)

                    diff = torch.norm(T - T_prev, p='fro')
                    if diff < 1e-7:
                        print(f"Converged at iter {t}, diff={diff.item()}")
                        break
                print('gw discrepancy finish!')
                torch.save(T, "output/T_gw.pt")
            else:
                T = torch.load("output/T_gw.pt")
                print('load gene-disease OT........')
            cost = cost_st - 2 * torch.matmul(torch.matmul(cost_s, T), torch.t(cost_t))
        else:
            # f1(a) = a*log(a) - a, f2(b) = b, h1(a) = a, h2(b) = log(b)
            # cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
            # cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
            f1_st = torch.matmul(cost_s * torch.log(cost_s + 1e-5) - cost_s, mu_s).repeat(1, nt)
            f2_st = torch.matmul(torch.t(mu_t), torch.t(cost_t)).repeat(ns, 1)
            cost_st = f1_st + f2_st
            # ff= True
            if not self.has_ot_matrix:
                T_prev = T.clone()
                for t in range(self.outer_iter):
                    cost = cost_st - torch.matmul(torch.matmul(cost_s, T), torch.t(torch.log(cost_t + 1e-5)))
                    with torch.no_grad():
                        T = self.sinkhorn_iter(mu_s, mu_t, cost)

                    diff = torch.norm(T - T_prev, p='fro')
                    if diff < 1e-7:
                        print(f"Converged at iter {t}, diff={diff.item()}")
                        break
                print('gw discrepancy finish!')
                torch.save(T, "output/T_gw_KL.pt")
            else:
                T = torch.load("output/T_gw_KL.pt")
            cost = cost_st - torch.matmul(torch.matmul(cost_s, T), torch.t(torch.log(cost_t + 1e-5)))

        d_gw = (cost * T).sum()
        return T, d_gw, cost

    def gda_prediction(self):

        gene_names = list(self.gene_graph.nodes())
        dis_names = list(self.dis_graph.nodes())
        dis_graph_ex, dis_edge_label = self.disease_disease_ot(self.has_ot_matrix, self.dis_graph,
                                                               self.dis_feat, self.device)
        gene_gw_cost = self.self_cost_mat(self.gene_feat, cost_type=self.cost_type, device=self.device)
        dis_gw_cost = self.self_cost_mat(self.dis_feat, cost_type=self.cost_type, device=self.device)

        T_gw, d_gw, cost_gw = self.gromov_wasserstein_discrepancy(gene_gw_cost, dis_gw_cost, loss_type=self.loss_type)
        threshold = np.quantile(T_gw.cpu().numpy(), 0.01)
        neg_mask = (T_gw < threshold)
        print(f'disease2gene graph  {neg_mask.sum()} negative edges')
        neg_pairs = []

        for gi, di in zip(*neg_mask.nonzero(as_tuple=True)):
            gi_idx = gi.item()
            di_idx = di.item()
            neg_pairs.append((gene_names[gi_idx], dis_names[di_idx]))

        dis2gene = self.dis2gene[self.dis2gene["geneSymbol"].isin(set(gene_names))]
        num_gene_nodes = dis2gene["geneSymbol"].nunique()
        num_disease_nodes = dis2gene["DOID"].nunique()
        print(f'DisGeNet gene nodes number:{num_gene_nodes}, disease nodes number:{num_disease_nodes}'
              f', total nodes number:{num_disease_nodes+num_gene_nodes}')
        # dis2gene.to_csv('data/Dis2Gene_filter.csv')

        pos_samples = set(zip(dis2gene["geneSymbol"], dis2gene["DOID"]))

        # get negative samples
        neg_samples = [pair for pair in neg_pairs if pair not in pos_samples]

        # random sampling pos and neg samples
        random.seed(self.random_seed)
        pos_pairs = random.sample(list(pos_samples), self.sample_num)
        neg_pairs = random.sample(neg_samples, self.sample_num)
        print(len(neg_pairs))

        # split train test data
        pos_test_size = int(0.2 * len(pos_pairs))
        neg_test_size = int(0.2 * len(neg_pairs))
        train_pos_pairs = pos_pairs[pos_test_size:]
        train_neg_pairs = neg_pairs[neg_test_size:]
        test_pos_pairs = pos_pairs[:pos_test_size]
        test_neg_pairs = neg_pairs[:neg_test_size]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Disease graph: get edge_index
        dis_nodelist = list(dis_graph_ex.nodes())
        dis_node2idx = {n: i for i, n in enumerate(dis_nodelist)}

        dis_edges = []
        for u, v in self.dis_graph.edges():
            dis_edges.append([dis_node2idx[u], dis_node2idx[v]])

        dis_edge_idx = torch.tensor(dis_edges, dtype=torch.long, device=self.device).t().contiguous()

        # Gene graph: get edge_index
        gene_nodelist = list(self.gene_graph.nodes())
        gene_node2idx = {n: i for i, n in enumerate(gene_nodelist)}

        gene_edges = []
        for u, v in self.gene_graph.edges():
            gene_edges.append([gene_node2idx[u], gene_node2idx[v]])

        gene_edge_idx = torch.tensor(gene_edges, dtype=torch.long, device=self.device).t().contiguous()

        # trans samples name to index
        train_pos_idx = self.pairs_to_index(train_pos_pairs, gene_node2idx, dis_node2idx)
        train_neg_idx = self.pairs_to_index(train_neg_pairs, gene_node2idx, dis_node2idx)

        # GCN dim
        dis_in = self.dis_feat.shape[1]
        dis_out = int(dis_in / 8)
        gene_in = self.gene_feat.shape[1]
        gene_out = int(gene_in / 8)

        # cross validation
        # for fold, (train_idx, val_idx) in enumerate(kf.split(train_pos_pairs)):
        #
        #     print(f"Fold {fold + 1}, training GCN model...")
        #
        #     # init GCN model
        #     dis_gcn = GCNConv(in_channels=dis_in, out_channels=dis_out).to(self.device)
        #     gene_gcn = GCNConv(in_channels=gene_in, out_channels=gene_out).to(self.device)
        #
        #     # init GCN optimizer
        #     optimizer = torch.optim.Adam(
        #         list(gene_gcn.parameters()) + list(dis_gcn.parameters()),
        #         lr=self.lr
        #     )
        #
        #     # training pairs
        #     tr_pos_idx = [train_pos_idx[i] for i in train_idx]
        #     tr_neg_idx = [train_neg_idx[i] for i in train_idx]
        #     tr_pos_pairs = [train_pos_pairs[i] for i in train_idx]
        #     tr_neg_pairs = [train_neg_pairs[i] for i in train_idx]
        #
        #     # val pairs
        #     val_pos_pairs = [train_pos_pairs[i] for i in val_idx]
        #     val_neg_pairs = [train_neg_pairs[i] for i in val_idx]
        #
        #     gene_gcn, dis_gcn = self.train_gcn(gene_gcn, dis_gcn, gene_edge_idx,
        #                                        dis_edge_idx, tr_pos_idx, tr_neg_idx)
        #
        #     print("Training classifier...")
        #
        #     gene_emb = gene_gcn(self.gene_feat.to(self.device), gene_edge_idx).detach()
        #     dis_emb = dis_gcn(self.dis_feat.to(self.device), dis_edge_idx).detach()
        #
        #     def embed_pairs(pairs):
        #         idx = self.pairs_to_index(pairs, gene_node2idx, dis_node2idx)
        #         g = torch.stack([gene_emb[g] for g, d in idx])
        #         d = torch.stack([dis_emb[d] for g, d in idx])
        #         return g, d
        #
        #     g_pos_emb, d_pos_emb = embed_pairs(tr_pos_pairs)
        #     g_neg_emb, d_neg_emb = embed_pairs(tr_neg_pairs)
        #
        #     X_train = torch.cat([torch.cat([g_pos_emb, d_pos_emb], dim=1),
        #                          torch.cat([g_neg_emb, d_neg_emb], dim=1)], dim=0)
        #     y_train = torch.cat([torch.ones(len(g_pos_emb)), torch.zeros(len(g_neg_emb))]).unsqueeze(1)
        #
        #     # validation set
        #     vg_pos_emb, vd_pos_emb = embed_pairs(val_pos_pairs)
        #     vg_neg_emb, vd_neg_emb = embed_pairs(val_neg_pairs)
        #     X_val = torch.cat([torch.cat([vg_pos_emb, vd_pos_emb], dim=1),
        #                         torch.cat([vg_neg_emb, vd_neg_emb], dim=1)], dim=0)
        #     y_val = torch.cat([torch.ones(len(vg_pos_emb)), torch.zeros(len(vg_neg_emb))]).unsqueeze(1)
        #
        #     # training classifier
        #     mlp = MLP(gene_out).to(self.device)
        #     optim = torch.optim.Adam(mlp.parameters(), lr=self.lr)
        #     loss_fn = nn.BCELoss()
        #
        #     loader = DataLoader(
        #         TensorDataset(X_train.to(self.device), y_train.to(self.device)),
        #         batch_size=256, shuffle=True
        #     )
        #
        #     for epoch in range(self.MLP_epochs):
        #         for xb, yb in loader:
        #             pred = mlp(xb[:, :gene_out], xb[:, gene_out:])
        #             loss = loss_fn(pred, yb)
        #             optim.zero_grad()
        #             loss.backward()
        #             optim.step()
        #         print(f"Classifier epoch {epoch} | loss={loss.item():.4f}")
        #
        #     # validation
        #     with torch.no_grad():
        #         pred = mlp(X_val.to(self.device)[:, :gene_out], X_val.to(self.device)[:, gene_out:])
        #         # acc = ((pred > 0.5) == y_test.to(self.device)).float().mean()
        #         y_true = y_val.cpu().numpy()
        #         acc, auc, f1, aupr = self.evaluate_metrics(y_true, pred)
        #
        #     self.accs.append(acc)
        #     self.aucs.append(auc)
        #     self.f1_scores.append(f1)
        #     self.auprs.append(aupr)
        #
        #     print(f"Fold {fold + 1} validation acc = {acc:.4f}, auc = {auc:.4f}, aupr = {aupr:.4f} f1:{f1:.4f}")
        #
        # print(f'Final GDA validation accuracies : {np.mean(self.accs):.4f} ± {np.std(self.accs):.4f}, '
        #       f'aucs : {np.mean(self.aucs):.4f} ± {np.std(self.aucs):.4f}, '
        #       f'auprs : {np.mean(self.auprs):.4f} ± {np.std(self.auprs):.4f}, '
        #       f'f1 scores : {np.mean(self.f1_scores):.4f} ± {np.std(self.f1_scores):.4f}')

        # test---------------
        print('start test..........')
        # init GCN model
        dis_gcn = GCNConv(in_channels=dis_in, out_channels=dis_out).to(self.device)
        gene_gcn = GCNConv(in_channels=gene_in, out_channels=gene_out).to(self.device)

        # training gcn model
        gene_gcn, dis_gcn = self.train_gcn(gene_gcn, dis_gcn, gene_edge_idx,
                                           dis_edge_idx, train_pos_idx, train_neg_idx, loss_curve=True)

        print("Training classifier...")
        dis_emb = dis_gcn(self.dis_feat.to(self.device), dis_edge_idx).detach()
        gene_emb = gene_gcn(self.gene_feat.to(self.device), gene_edge_idx).detach()

        def embed_pairs(pairs):
            idx = self.pairs_to_index(pairs, gene_node2idx, dis_node2idx)
            g = torch.stack([gene_emb[g] for g, d in idx])
            d = torch.stack([dis_emb[d] for g, d in idx])
            return g, d

        g_pos_emb, d_pos_emb = embed_pairs(train_pos_pairs)
        g_neg_emb, d_neg_emb = embed_pairs(train_neg_pairs)

        X_train = torch.cat([torch.cat([g_pos_emb, d_pos_emb], dim=1),
                             torch.cat([g_neg_emb, d_neg_emb], dim=1)], dim=0)
        y_train = torch.cat([torch.ones(len(g_pos_emb)), torch.zeros(len(g_neg_emb))]).unsqueeze(1)

        # test set
        teg_pos_emb, ted_pos_emb = embed_pairs(test_pos_pairs)
        teg_neg_emb, ted_neg_emb = embed_pairs(test_neg_pairs)
        X_test = torch.cat([torch.cat([teg_pos_emb, ted_pos_emb], dim=1),
                           torch.cat([teg_neg_emb, ted_neg_emb], dim=1)], dim=0)
        y_test = torch.cat([torch.ones(len(teg_pos_emb)), torch.zeros(len(teg_neg_emb))]).unsqueeze(1)

        # save test data
        # os.makedirs('output/save_test_data'), exist_ok=True)
        # torch.save({
        #     "X_test": X_test,
        #     "y_test": y_test
        # }, f"output/save_test_data/test_data_lr_{self.lr}_alpha_{self.alpha}_beta_{self.beta}_margin_{self.margin}_random_{self.random_seed}.pt")

        # training classifier
        mlp = MLP(gene_out).to(self.device)
        optim = torch.optim.Adam(mlp.parameters(), lr=self.lr, weight_decay=1e-3)
        loss_fn = nn.BCELoss()

        loader = DataLoader(
            TensorDataset(X_train.to(self.device), y_train.to(self.device)),
            batch_size=256, shuffle=True
        )

        for epoch in range(self.MLP_epochs):
            epoch_loss = 0
            batch_count = 0

            for xb, yb in loader:
                pred = mlp(xb[:, :gene_out], xb[:, gene_out:])
                loss = loss_fn(pred, yb)
                epoch_loss += loss.item()
                batch_count += 1
                optim.zero_grad()
                loss.backward()
                optim.step()
            epoch_loss /= batch_count
            self.mlp_train_loss.append(epoch_loss*10)
            print(f"Classifier epoch {epoch} | loss={epoch_loss*10:.4f}")

            with torch.no_grad():
                pred = mlp(X_test.to(self.device)[:, :gene_out], X_test.to(self.device)[:, gene_out:])
                y_true = y_test.cpu().numpy()
                acc, auc, f1, aupr = self.evaluate_metrics(y_true, pred)
                self.epoch_aucs.append(auc)

        # test results
        with torch.no_grad():
            pred = mlp(X_test.to(self.device)[:, :gene_out], X_test.to(self.device)[:, gene_out:])
            y_true = y_test.cpu().numpy()
            acc, auc, f1, aupr = self.evaluate_metrics(y_true, pred)

        # plot loss curve
        # os.makedirs('output/loss_curve', exist_ok=True)
        curve_path = f'output/loss_curve/all_loss_lr_{self.lr}_alpha_{self.alpha}_beta_{self.beta}_margin_{self.margin}_random_{self.random_seed}.png'
        # self.plot_loss_curve(curve_path)

        # save model
        # os.makedirs('output/save_model', exist_ok=True)
        # model_path = f'output/save_model/model_lr_{self.lr}_alpha_{self.alpha}_beta_{self.beta}_margin_{self.margin}_random_{self.random_seed}_all.pt'
        # torch.save({
        #     "dis_gcn": dis_gcn.state_dict(),
        #     "gene_gcn": gene_gcn.state_dict(),
        #     "classifier": mlp.state_dict(),
        # }, model_path)
        # print(f"Model saved to {model_path}")

        print(f"test acc:{acc:.4f}, auc:{auc:.4f}, aupr:{aupr:.4f} f1:{f1:.4f}")

    def train_gcn(self, gene_gcn, dis_gcn, gene_edge_idx, dis_edge_idx, train_pos, train_neg, loss_curve=False):

        # training model
        for epoch in range(self.GCN_epochs):
            gene_embedding = gene_gcn(self.gene_feat.to(self.device), gene_edge_idx)
            dis_embedding = dis_gcn(self.dis_feat.to(self.device), dis_edge_idx)
            # positive pairs embeddings
            g_pos = torch.stack([gene_embedding[g] for g, d in train_pos], dim=0)
            d_pos = torch.stack([dis_embedding[d] for g, d in train_pos], dim=0)

            # negative pairs embeddings
            g_neg = torch.stack([gene_embedding[g] for g, d in train_neg], dim=0)
            d_neg = torch.stack([dis_embedding[d] for g, d in train_neg], dim=0)

            # init GCN optimizer
            optimizer = torch.optim.Adam(
                list(gene_gcn.parameters()) + list(dis_gcn.parameters()),
                lr=self.lr,
                weight_decay=1e-3
            )

            # loss
            pos_dist = torch.norm(g_pos - d_pos, p=2, dim=1)
            neg_dist = torch.norm(g_neg - d_neg, p=2, dim=1)

            loss = pos_dist.mean() + self.alpha * torch.relu(self.margin - neg_dist).mean()
            if loss_curve:
                self.gcn_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} | GCN loss = {loss.item():.4f}")

        return gene_gcn, dis_gcn

    def disease_disease_ot(self, ot_matrix, dis_graph, dis_feature, device):

        edge_feat, edge_label, edge_index = self.get_edge_embeddings(dis_graph, dis_feature,
                                                                     k=5, nodelist=None)
        dis_nodes = list(dis_graph.nodes())
        if not ot_matrix:

            P = torch.tensor(edge_feat[edge_label == 1], dtype=torch.float32, device=device)
            U = torch.tensor(edge_feat[edge_label == 0], dtype=torch.float32, device=device)
            dis_w_cost = self.mutual_cost_mat(P, U, cost_type=self.cost_type, device=device)
            dis_cost = dis_w_cost / dis_w_cost.std()
            with torch.no_grad():
                T = self.wasserstein_distance(P, U, dis_cost, device)
                os.makedirs('output/disease', exist_ok=True)
                np.save("output/disease/T_matrix.npy", T.cpu().numpy())
                print('wasserstein distance finished!')
        else:
            T = torch.tensor(np.load("output/disease/T_matrix.npy"), dtype=torch.float32, device=device)
            print('load disease-disease OT.........')

        T_max, idx = T.max(dim=0)  # column max
        sigma = min(1.0, 1.0 / (self.beta * len(edge_feat[edge_label == 1])))
        P_hat = T_max >= sigma
        a = P_hat.sum().item()
        print(f'disease graph add {a} new edges')
        unlabeled_idx = np.where(edge_label == 0)[0]

        # Select potentially positive samples from the unlabeled samples.
        selected_idx = unlabeled_idx[P_hat.cpu().numpy()]
        edge_label[selected_idx] = 1
        selected_edges = edge_index[selected_idx]

        dis_graph_ex = copy.deepcopy(dis_graph)

        for u, v in selected_edges:
            real_u = dis_nodes[u]
            real_v = dis_nodes[v]
            dis_graph_ex.add_edge(real_u, real_v)

        return dis_graph_ex, torch.tensor(edge_label, dtype=torch.float32)

    def evaluate_metrics(self, y_true, pred):

        y_pred = (pred > 0.5).int().cpu().numpy()
        y_prob = pred.detach().cpu().numpy()

        # evaluation metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        aupr = average_precision_score(y_true, y_prob)

        return acc, auc, f1, aupr

    def plot_loss_curve(self, curve_path):

        plt.figure(figsize=(10, 6))

        gcn_epochs = range(1, len(self.gcn_train_loss) + 1)
        mlp_epochs = range(1, len(self.mlp_train_loss) + 1)
        auc_epochs = range(1, len(self.epoch_aucs) + 1)

        ax1 = plt.gca()
        ax1.plot(gcn_epochs, self.gcn_train_loss, label="GCN Train Loss", linewidth=2)
        ax1.plot(mlp_epochs, self.mlp_train_loss, label="MLP Train Loss", linewidth=2)

        ax1.set_xlabel("Epoch", fontsize=14)
        ax1.set_ylabel("Loss", fontsize=14)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(
            auc_epochs,
            self.epoch_aucs,
            label="Test AUC",
            linewidth=2,
            linestyle="--"
        )
        ax2.set_ylabel("AUC", fontsize=14)
        ax2.tick_params(axis='y', labelsize=12)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        plt.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper right",
            fontsize=12
        )

        plt.title("Loss Curves and Test AUC", fontsize=16)
        plt.tight_layout()
        plt.savefig(curve_path, dpi=400)

    def load_model(self):

        # load test data
        data = torch.load(f"output/save_test_data/test_data_lr_{self.lr}_beta_{self.beta}_alpha_{self.alpha}_random_{self.random_seed}.pt")
        X_test = data["X_test"]
        y_test = data["y_test"]

        # load model
        gene_in = self.gene_feat.shape[1]
        gene_out = int(gene_in / 8)
        gda_model = torch.load(f'output/save_model/model_lr_{self.lr}_beta_{self.beta}_alpha_{self.alpha}_random_{self.random_seed}.pt')
        mlp = MLP(gene_out).to(self.device)

        mlp.load_state_dict(gda_model["classifier"])

        # test phrase
        with torch.no_grad():
            pred = mlp(X_test.to(self.device)[:, :gene_out], X_test.to(self.device)[:, gene_out:])
            y_true = y_test.cpu().numpy()
            acc, auc, f1, aupr = self.evaluate_metrics(y_true, pred)

        print(f'parameters lr_{self.lr}_beta_{self.beta}_alpha_{self.alpha}_random_{self.random_seed}')
        print(f"test acc:{acc:.4f}, auc:{auc:.4f}, f1:{f1:.4f}, aupr:{aupr:.4f}")


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1)
        )

    def forward(self, g_emb, d_emb):
        x = torch.cat([g_emb, d_emb], dim=1)
        x = self.fc(x)
        return torch.sigmoid(x)