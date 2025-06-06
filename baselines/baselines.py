import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch_geometric.utils import degree, to_undirected, softmax, to_scipy_sparse_matrix
from scipy.special import logsumexp
from numpy.linalg import norm, pinv
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from torch_sparse import SparseTensor, matmul
import numpy as np
import faiss
from scipy.stats import entropy

def propagation(e, edge_index, alpha=0, K=8):
    e = e.unsqueeze(1)
    N = e.shape[0]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm = 1. / d[col]
    value = torch.ones_like(row) * d_norm
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    adj = adj.to_device(e.device)
    for _ in range(K):
        e = e * alpha + matmul(adj, e) * (1 - alpha)
    
    return e.squeeze(1)


class Entropy():
    def __init__(self, args):
        self.dataset = args.dataset

    def inference(self, logits):
        pred = F.softmax(logits, dim=1).detach().cpu()
        ENT = entropy(pred, axis=1)
        return pred, -torch.tensor(ENT).cuda()

    def detect(self, logits):
        if self.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            pass
        else: # for single-label multi-class classification
            pred, score = self.inference(logits)
        return score

class MSP():
    def __init__(self, args):
        self.dataset = args.dataset

    def inference(self, logits):
        sp = torch.softmax(logits, dim=-1)
        score, pred = sp.max(dim=-1)
        return pred, score

    def detect(self, logits):
        if self.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            pass
        else: # for single-label multi-class classification
            pred, score = self.inference(logits)
        return score
        

class Energy():
    def __init__(self, args):
        self.T = args.T
        self.dataset = args.dataset

    def inference(self, logits):
        _, pred = torch.max(logits, dim=1)
        conf = self.T * torch.logsumexp(logits / self.T, dim=-1)
        return pred, conf

    def detect(self, logits):
        if self.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            pass
        else: # for single-label multi-class classification
            _, neg_energy = self.inference(logits)
        return neg_energy
    
class ODIN():
    def __init__(self, args) -> None:
        super().__init__()
        self.temperature = 1000
        self.noise = args.noise #0.0014
    
    def inference(self, logits):
        sp = torch.softmax(logits / self.temperature, dim=-1)
        score, pred = sp.max(dim=-1)
        return pred, score
    
    def detect(self, logits):
        _, neg_energy = self.inference(logits)
        return neg_energy

    
class KNN():
    def __init__(self, args) -> None:
        super().__init__()
        self.K = args.neighbors
        self.activation_log = None
        self.normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

    def setup(self, net: nn.Module, dataset_ind, train_idx, device):
        net.eval()
        x, edge_index = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        
        with torch.no_grad():
            feature = net(x, edge_index)
            self.train_feature = feature

        self.activation_log = self.normalizer(feature.data.cpu().numpy())
        self.index = faiss.IndexFlatL2(feature.shape[1])
        self.index.add(self.activation_log)

    @torch.no_grad()
    def detect(self, logit, train_idx):
        feature = logit
        # setup index
        feature_normed = self.normalizer(feature.cpu().numpy())
        self.index = faiss.IndexFlatL2(feature.shape[1])
        self.index.add(feature_normed[train_idx])
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        kth_dist = torch.from_numpy(kth_dist)
        return kth_dist
    
class GNNSafe():
    def __init__(self, args):
        self.T = args.T
        self.dataset = args.dataset

    def inference(self, logits):
        _, pred = torch.max(logits, dim=1)
        conf = self.T * torch.logsumexp(logits / self.T, dim=-1)
        return pred, conf

    def detect(self, logits, edge_index, args):
        '''return negative energy, a vector for all input nodes'''
        if self.dataset in ('proteins', 'ppi'): # for multi-label binary classification
            pass
        else: # for single-label multi-class classification
            _, scores = self.inference(logits)
        scores = propagation(scores, edge_index, alpha=args.alpha, K=args.K)
        return scores

class GRASP():
    def __init__(self, args):
        self.T = args.T
        self.dataset = args.dataset

    def inference(self, logits, score='MSP'):
        if score == 'Energy':
            _, pred = torch.max(logits, dim=1)
            score = self.T * torch.logsumexp(logits / self.T, dim=-1)
        elif score == 'MSP':
            sp = torch.softmax(logits, dim=-1)
            score, pred = sp.max(dim=-1)
        return pred, score

    def detect(self, logits, dataset_ind, train_idx, test_id, test_ood, args):
        _, scores = self.inference(logits)
        test_nodes = torch.concat([test_id, test_ood])
        row, col = dataset_ind.edge_index
        if args.col: row,col=col,row
        N = dataset_ind.num_nodes
        value = torch.ones_like(row)
        device = logits.device

        adj1 = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N, N))
        adj1 = adj1.to_device(device)
        add_nodes = select_G(scores, train_idx, test_nodes, adj1, args)
        scores[train_idx] = torch.where(scores[train_idx]<1, 1., scores[train_idx])

        edge_index = to_undirected(dataset_ind.edge_index)
        row, col = edge_index
        d = degree(col, N).float()
        d_add = torch.zeros(N, dtype=d.dtype, device=d.device)
        d_add[add_nodes] = len(add_nodes)
        d += d_add
        d_inv = 1. / d.unsqueeze(1)
        d_inv = torch.nan_to_num(d_inv, nan=0.0, posinf=0.0, neginf=0.0)
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        adj = adj.to_device(device)
        d_inv = d_inv.type(scores.dtype)
        d_inv = d_inv.to(device)
        e_add = torch.zeros(N, 1, dtype=scores.dtype, device=device)
        scores = scores.unsqueeze(1)
        for k in range(1, args.K+1):
            e_add[add_nodes] = scores[add_nodes].sum()*d_inv[add_nodes]
            scores = scores * args.alpha + (matmul(adj, scores)+args.delta*e_add) * (1 - args.alpha)
            scores[train_idx] = torch.where(scores[train_idx]<1, 1., scores[train_idx])
            if args.adj1 and k < args.K:
                add_nodes = select_G2(scores, train_idx, test_nodes, adj1, args, k)
    
        return scores.squeeze(1) 
    
def select_G2(scores, train_idx, test_nodes, adj, args, k):
    if args.test:
        nodes_use = test_nodes
    else:
        nodes_use = train_idx
    if args.tau2 == 100:
        return nodes_use.tolist()
    K = int(args.tau2/100 * len(nodes_use))
    if args.st == 'random':
        return np.random.choice(nodes_use, K, replace=False).tolist()

    scores = scores.squeeze(1)
    values = scores[test_nodes].cpu()
    if args.st == 'test':
        K = int(args.tau2/100 * len(test_nodes))
        return test_nodes[np.argpartition(values, kth=-K)[-K:]].tolist()

    #get Sid and Sood
    p = args.tau1
    thresholds1 = np.percentile(values, p)
    mask = values < thresholds1
    sood = test_nodes[mask]
    thresholds2 = np.percentile(values, 100-p)
    mask = values > thresholds2
    sid = test_nodes[mask]

    #calculate metric to select G
    N = scores.size(0)
    id_count = torch.zeros(N)
    ood_count = torch.zeros(N)
    id_count[sid] = 1
    ood_count[sood] = 1
    device = scores.device
    id_count = id_count.unsqueeze(1).to(device)
    ood_count = ood_count.unsqueeze(1).to(device)
    id_add = torch.zeros(N, 1, dtype=scores.dtype, device=device)
    ood_add = torch.zeros(N, 1, dtype=scores.dtype, device=device)

    for _ in range(k+1):
        id_count = matmul(adj, id_count)
        ood_count = matmul(adj, ood_count)

    id_count = id_count.squeeze(1).cpu()
    ood_count = ood_count.squeeze(1).cpu()

    metrics = id_count[nodes_use]/(ood_count[nodes_use]+1)
    
    #select the top big K
    if args.st == 'top':
        return nodes_use[np.argpartition(metrics, kth=-K)[-K:]].tolist()
    #select the top small K
    elif args.st == 'low':
        return nodes_use[np.argpartition(metrics, kth=K)[: K]].tolist()
    
def select_G(scores, train_idx, test_nodes, adj, args):
    if args.tau2 == 100:
        return train_idx.tolist()
    K = int(args.tau2/100 * len(train_idx))
    if args.st == 'random':
        return np.random.choice(train_idx, K, replace=False).tolist()

    values = scores[test_nodes].cpu()
    if args.st == 'test':
        K = int(args.tau2/100 * len(test_nodes))
        return test_nodes[np.argpartition(values, kth=-K)[-K:]].tolist()

    #get Sid and Sood
    p = args.tau1
    thresholds1 = np.percentile(values, p)
    mask = values < thresholds1
    sood = test_nodes[mask]
    thresholds2 = np.percentile(values, 100-p)
    mask = values > thresholds2
    sid = test_nodes[mask]

    #calculate metric to select G
    N = scores.size(0)
    id_count = torch.zeros(N)
    ood_count = torch.zeros(N)
    id_count[sid] = 1
    ood_count[sood] = 1
    device = scores.device
    id_count = id_count.unsqueeze(1).to(device)
    ood_count = ood_count.unsqueeze(1).to(device)
    id_count = matmul(adj, id_count)
    ood_count = matmul(adj, ood_count)

    id_count = id_count.squeeze(1).cpu()
    ood_count = ood_count.squeeze(1).cpu()

    metrics = id_count[train_idx]/(ood_count[train_idx]+1)
    
    #select the top big K
    if args.st == 'top':
        return train_idx[np.argpartition(metrics, kth=-K)[-K:]].tolist()
    #select the top small K
    elif args.st == 'low':
        return train_idx[np.argpartition(metrics, kth=K)[: K]].tolist()

