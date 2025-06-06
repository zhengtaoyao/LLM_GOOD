import sys
import argparse
import torch
from scipy.stats import entropy
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures, Compose
from torch_geometric.utils import homophily
sys.path.append("../..")
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import get_acc
from OOD.utils import EntropyLoss, get_consistent_loss_new, cosine_similarity, CE_uniform, seed_torch
from data_process import generate_masks_OOD
from sampling_methods import *
import numpy as np
from baselines.baselines import *

from basic_gnns import GCNDetector
from basic_gnns import MLP, GCNNet, GCNDetector
from torch_geometric.utils import to_dense_adj
import warnings
warnings.filterwarnings("ignore")
import math

import torch_geometric.data as Data
from sklearn import metrics

def train(args):
    set_random_seed(args.random_seed_data)

    # dataset parameters
    ID_classes = args.ID_classes
    splits = args.splits
    n_samples_init = splits[0]
    n_samples_per_class = splits[1]
    val_size_per_class = splits[2]
    test_size = splits[3]
    
    data_raw = torch.load("cora_random_sbert.pt", map_location='cpu')
    data_gpt = torch.load("gpt_llm_pt_file/cora_gpt4o_final.pt", map_location='cpu')

    # data_raw = torch.load("pubmed_random_sbert.pt", map_location='cpu')
    # data_gpt = torch.load("gpt_llm_pt_file/pubmed_gpt4o_final.pt", map_location='cpu')

    # data_raw = torch.load("citeseer_random_sbert.pt", map_location='cpu')
    # data_gpt = torch.load("gpt_llm_pt_file/citeseer_gpt4o_final.pt", map_location='cpu')

    # data_raw = torch.load("wikics_fixed_sbert.pt", map_location='cpu')
    # data_gpt = torch.load("gpt_llm_pt_file/wikics_gpt4o_final.pt", map_location='cpu')
    
    sudo_labels = data_gpt['pred'].cuda()
    detection_llm = torch.where(sudo_labels == -2, torch.tensor(0, device=sudo_labels.device), torch.tensor(1, device=sudo_labels.device))

    matching_indices = [idx for idx, label in enumerate(sudo_labels) if label in ID_classes]
    import copy
    data = copy.deepcopy(data_raw)
    
    #calculate the number of nodes in each class
    node_labels = data.y
    class_counts = {cls: 0 for cls in ID_classes}
    for cls in ID_classes:
        class_counts[cls] = (node_labels == cls).sum().item()
    for cls, count in class_counts.items():
        print(f'Class {cls}: {count} nodes')
    
    ID_mask, OOD_mask, train_mask_ID, val_mask_ID, test_mask_ID, train_mask_OOD, val_mask_OOD, test_mask_OOD, \
    detection_mask_val, detection_y_val, joint_y_val, detection_mask_test, detection_y_test, joint_y_test, ID_y, left_idx_all, detection_mask_train, joint_y_train, joint_y_all = \
        generate_masks_OOD(data, n_samples_init, ID_classes, n_samples_per_class, val_size_per_class, test_size, args.random_seed_data)

    assert (ID_mask | OOD_mask).sum() == data.num_nodes
    assert (train_mask_ID | val_mask_ID | test_mask_ID | train_mask_OOD | val_mask_OOD | test_mask_OOD).sum() == \
           train_mask_ID.sum() + val_mask_ID.sum() + test_mask_ID.sum() + train_mask_OOD.sum() + val_mask_OOD.sum() + test_mask_OOD.sum()
    assert detection_mask_val.sum() == len(detection_y_val) == len(joint_y_val)
    assert detection_mask_test.sum() == len(detection_y_test) == len(joint_y_test)
    # assert train_mask_ID.sum() == train_mask_OOD.sum() == n_samples_per_class * len(ID_classes)
    assert val_mask_ID.sum() == val_mask_OOD.sum() == val_size_per_class * len(ID_classes)
    assert test_mask_ID.sum() == test_mask_OOD.sum() == test_size

    data.train_mask, data.val_mask, data.test_mask = train_mask_ID, val_mask_ID, test_mask_ID
    data.y = ID_y
    data.y_origin = data.y
    data.joint_y_train, data.detection_mask_train = torch.tensor(joint_y_train), detection_mask_train
    data.joint_y_test, data.detection_mask_test = torch.tensor(joint_y_test), detection_mask_test
    data.joint_y_val, data.detection_mask_val = torch.tensor(joint_y_val), detection_mask_val
    data.joint_y_all = torch.tensor(joint_y_all)
    
    print('ID size: {}, OOD size: {}, total size: {}.'.format(ID_mask.sum(), OOD_mask.sum(), data.num_nodes))
    print('train%ID: {:.2%}, val%ID: {:.2%}, test%ID: {:.2%}.'.format(train_mask_ID.sum() / ID_mask.sum(),
                                                                      val_mask_ID.sum() / ID_mask.sum(),
                                                                      test_mask_ID.sum() / ID_mask.sum()))
    print('train%OOD: {:.2%}, val%OOD: {:.2%}, test%OOD: {:.2%}.'.format(train_mask_OOD.sum() / OOD_mask.sum(),
                                                                         val_mask_OOD.sum() / OOD_mask.sum(),
                                                                         test_mask_OOD.sum() / OOD_mask.sum()))
    device = torch.device('cuda')
    data = data.to(device)


    def init():
        # init model
        # seed_torch(args.random_seed_model)
        in_dim = data.x.shape[1]
        out_dim = len(ID_classes)
        if args.model_name == 'MLP':
            model = MLP(in_dim, args.hidden_dim, out_dim, args.drop_prob).to(device)
        elif args.model_name == 'GCN':
            model = GCNNet(in_dim, args.hidden_dim, out_dim, args.drop_prob, bias=True).to(device)
            model2 = GCNDetector(in_dim, args.hidden_dim, out_dim, args.drop_prob, bias=True).to(device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        return model, model2, optimizer, optimizer2


    def evaluate_ood(iid_score, ood_score, tpr=0.95):
        iid_score, ood_score = iid_score.cpu(), ood_score.cpu()
        scores = torch.cat([iid_score, ood_score]).numpy()
        y_true = np.zeros(len(scores))
        y_true[:len(iid_score)] = 1
        fpr_list, tpr_list, threshold_list = metrics.roc_curve(y_true, scores)
        fpr = fpr_list[np.argmax(tpr_list >= tpr)]
        thresh = threshold_list[np.argmax(tpr_list >= tpr)]
        auroc = metrics.auc(fpr_list, tpr_list)

        precision_in, recall_in, thresholds_in \
            = metrics.precision_recall_curve(y_true, scores)
        aupr_in = metrics.auc(recall_in, precision_in)
        return auroc, aupr_in, fpr, thresh

    #train GCN ID classifier
    def train(model, optimizer):
        # train
        ood = eval(args.ood)(args)

        xent = nn.CrossEntropyLoss()
        ent_loss_func = EntropyLoss(reduction=False)
        best_t = args.epochs - 1
        best_metric = 0
        patience = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = torch.zeros(1).to(device)
            embeds, logits = model(data)
            sup_loss = xent(logits[data.train_mask], data.y[data.train_mask])
            loss += sup_loss
            loss.backward()
            optimizer.step()
            # validate
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    embeds, logits = model(data)
                preds = logits.argmax(axis=1).detach()
                val_acc = get_acc(data.y, preds, data.val_mask)
                
                if args.ood in ['MSP', 'Energy', 'ODIN', 'Entropy']:
                    scores = ood.detect(logits)
                elif args.ood == 'GNNSafe':
                    scores = ood.detect(logits, data.edge_index, args)
                elif args.ood == 'GRASP':
                    train_id = torch.nonzero(train_mask_ID, as_tuple=True)[0]
                    val_id = torch.nonzero(val_mask_ID, as_tuple=True)[0]
                    test_id = torch.nonzero(test_mask_ID, as_tuple=True)[0]
                    test_ood = torch.nonzero(test_mask_OOD, as_tuple=True)[0]
                    scores = ood.detect(logits, data, torch.concat([train_id, val_id]), test_id, test_ood, args)

                labels = torch.tensor(detection_y_val, dtype=torch.int32)
                scores = scores[[data.detection_mask_val]].to(device)
                iid_score = scores[torch.where(labels==0)]
                ood_score = scores[torch.where(labels==1)]
                auroc, aupr_in, fpr = evaluate_ood(iid_score, ood_score)[:-1]

                current_metric = val_acc + auroc
                if  current_metric > best_metric:
                    best_t = epoch
                    patience = 0
                    best_metric = current_metric
                    torch.save(model.state_dict(), 'best_GNN.pkl')
                else:
                    patience += 1
                    if patience > 20:
                        break

        return best_metric, best_t


    #evaluate GCN ID classifier
    def evaluate(model):
        # evaluate
        ood = eval(args.ood)(args)

        with torch.no_grad():
            model.load_state_dict(torch.load('best_GNN.pkl'))
            model.eval()
            # classification
            embeds, logits = model(data)
            preds = logits.argmax(axis=1).detach()
            test_acc = get_acc(data.y, preds, data.test_mask)
            print('test_acc:{}'.format(test_acc.item()))

            if args.ood in ['MSP', 'Energy', 'ODIN', 'Entropy']:
                scores = ood.detect(logits)
            elif args.ood == 'GNNSafe':
                scores = ood.detect(logits, data.edge_index, args)
            elif args.ood == 'GRASP':
                    train_id = torch.nonzero(train_mask_ID, as_tuple=True)[0]
                    val_id = torch.nonzero(val_mask_ID, as_tuple=True)[0]
                    test_id = torch.nonzero(test_mask_ID, as_tuple=True)[0]
                    test_ood = torch.nonzero(test_mask_OOD, as_tuple=True)[0]
                    scores = ood.detect(logits, data, torch.concat([train_id, val_id]), test_id, test_ood, args)

            labels = torch.tensor(detection_y_test, dtype=torch.int32)
            scores = scores[[data.detection_mask_test]].to(device)
            iid_score = scores[torch.where(labels==0)]
            ood_score = scores[torch.where(labels==1)]
            
            auroc, aupr_in, fpr = evaluate_ood(iid_score, ood_score)[:-1]

        return embeds, logits, test_acc, auroc, aupr_in, fpr
    

    model, model2, opt, opt2 = init()    
    k = len(ID_classes) * args.splits[0]
    left_ID_indices = left_idx_all
    iter_num = 0
    adj = to_dense_adj(data.edge_index).squeeze(0)
    budget_ad = len(args.ID_classes) * (n_samples_per_class-n_samples_init)
    
    idx_cand_an = left_idx_all
    cluster_num = 48
    
    
    for iter in range(iter_num + 1):
        set_random_seed(args.random_seed_data)
        
        # filter OOD nodes with LLM
        list1 = matching_indices #llm's 2708-test#-val#'s nodes, choosing 15C from idx_can_an to get the real label
        
        #idx_can_an is the llm considering it is id
        list2 = idx_cand_an #real 2708-test#-val#'s nodes, choosing 15C from idx_can_an to get the real label
        intersection = list(set(list1).intersection(list2))
        preds_idx_ID = intersection
        idx_cand_an = torch.tensor(idx_cand_an)
        
        # # Node selection with active learning methods
        # idx_selected = query_medoids_spec_nent(adj, embeds, prob_nc, budget_ad, preds_idx_ID, cluster_num)
        # idx_selected = torch.tensor(idx_selected).cuda()
        
        # random node selection from potential ID node set
        preds_idx_ID = torch.tensor(preds_idx_ID)
        # Shuffle and select random elements
        random_indices = torch.randperm(preds_idx_ID.shape[0])[:budget_ad]
        random_elements = preds_idx_ID[random_indices]
        
        idx_selected = random_elements.cuda()
    
        # split ID and OOD nodes from selected nodes
        joint_y_selected = data.joint_y_all[idx_selected]
        mask = torch.eq(joint_y_selected, len(args.ID_classes))
        idx_selected_id = idx_selected[~mask.cuda()]
        idx_selected_ood = idx_selected[mask.cuda()]
        
        # # Update state
        # idx_train = torch.cat((idx_train, idx_selected_id))
        # idx_selected = idx_selected.cpu().tolist()  # Convert tensor to list
        # idx_cand_an = idx_cand_an.cpu().numpy().tolist()
        # idx_cand_an = list(set(idx_cand_an)-set(idx_selected))
        
        for i in range(len(data.y)):
            # if i in idx_train:
            if i in idx_selected_id:
                data.train_mask[i] = 1
                data.detection_mask_train[i] = 1
            if i in idx_selected_ood:
                data.detection_mask_train[i] = 1
        
        #train ID classifier        
        best_metric, best_t = train(model, opt)
        embeds, prob_nc, test_acc, auroc, aupr, fpr = evaluate(model)
        
        num_id = data.train_mask.sum()  #smaller thatn 15C
        num_total = data.detection_mask_train.sum() #15C
        print('Number of ID nodes used for training: {}!!'.format(num_id))
        print('Total number of nodes used: {}!!'.format(num_total))
        precision = num_id/num_total

    return test_acc, auroc, aupr, fpr, precision


def local_ent_loss(logits, att, n_id_classes, m=0.5):
    att_norm = F.sigmoid(torch.hstack([att[0], att[1]]).mean(axis=1)).detach()  # n-dim
    mask = torch.ge(att_norm - m, 0)
    ce_uni = CE_uniform(n_id_classes, reduction=False)
    ce = ce_uni(logits)  # N-dim
    if mask.sum() > 0:
        loss = ce[mask].mean()
    else:
        loss = 0

    return loss

def set_random_seed(seed):
    import os
    import random
    import numpy as np
    import torch

    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)     # Set seed for Python's random module
    torch.manual_seed(seed)  # Set seed for PyTorch (CPU)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set seed for CUDA
        torch.cuda.manual_seed_all(seed)  # Set seed for all CUDA devices
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable cuDNN's auto-tuner to avoid non-determinism
    
    # Optional: Set an environment variable for Python hash seed (if hash-based randomness is used)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM-GOOD')
    parser.add_argument('--dataset', default="Planetoid_Cora", type=str)
    
    # #below is for cora ood class index
    parser.add_argument('--ID_classes', default=[4, 2, 5, 6], type=list) #[1, 2, 3, 4]

    # #below is for pubmed ood class index
    # parser.add_argument('--ID_classes', default=[0,1], type=list) #[1, 2, 3, 4]

    #below is for citeseer ood class index
    # parser.add_argument('--ID_classes', default=[0,1,2], type=list) #[1, 2, 3, 4]

    #below is for wikics ood class index
    # parser.add_argument('--ID_classes', default=[5,1,6,4], type=list) #[1, 2, 3, 4]

    parser.add_argument('--splits', default=[0, 10, 10, 500], type=list)
    parser.add_argument('--continuous', default=False, type=bool)
    parser.add_argument('--model_name', default='GCN', type=str)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--drop_prob', default=0.5, type=float)
    parser.add_argument('--drop_input', default=0.5, type=float)
    parser.add_argument('--drop_edge', default=0.6, type=float)
    parser.add_argument('--heads', default=1, type=int)
    parser.add_argument('--w_consistent', default=2.0, type=float)
    parser.add_argument('--w_ent', default=0.05, type=float)
    parser.add_argument('--w_discrepancy', default=5e-3, type=float)
    parser.add_argument('--margin', default=0.6, type=float)
    parser.add_argument('--random_seed_data', default=123, type=int)
    parser.add_argument('--random_seed_model', default=456, type=int)
    
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--ood', type=str, default='GNNSafe')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    parser.add_argument('--K', type=int, default=8, help='number of layers for belief propagation')
    parser.add_argument('--alpha', type=float, default=0., help='weight for residual connection in propagation')
    parser.add_argument('--st', type=str, default='top', choices=['top', 'low', 'random', 'test'], help='what metric to use')
    parser.add_argument('--col', action='store_true', help='use col to count connections')
    parser.add_argument('--adj1', action='store_true')
    parser.add_argument('--tau1', type=float, default=5, help='threshold to determine s_id and s_ood')
    parser.add_argument('--tau2', type=float, default=50, help='threshold to select train nodes as G')
    parser.add_argument('--delta', type=float, default=1.001, help='weight for G')

    args = parser.parse_args()
    # train(args)
    n_runs = 5
    auroc_ENT_list, aupr_0_ENT_list, aupr_1_ENT_list, fprs_ENT_list, test_acc_list, precision_list = [], [], [], [], [], []
    for i in range(n_runs):
        args.random_seed_data = args.random_seed_data + 1
        set_random_seed(args.random_seed_data)
        
        test_acc, auroc, aupr, fprs,precision = train(args)

        test_acc_list.append(test_acc)
        auroc_ENT_list.append(auroc)
        aupr_0_ENT_list.append(aupr)
        # aupr_1_ENT_list.append(aupr_1_ENT)
        fprs_ENT_list.append(fprs)
        precision_list.append(precision) 
        
    
    # Compute means and variances
    test_acc_mean = torch.stack(test_acc_list).mean().cpu().item()
    test_acc_var = torch.stack(test_acc_list).std().cpu().item()

    auroc_mean = np.mean([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in auroc_ENT_list])
    auroc_var = np.std([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in auroc_ENT_list])

    aupr_0_mean = np.mean([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in aupr_0_ENT_list])
    aupr_0_var = np.std([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in aupr_0_ENT_list])

    fprs_mean = np.mean([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in fprs_ENT_list])
    fprs_var = np.std([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in fprs_ENT_list])

    precision_mean = np.mean([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in precision_list])
    precision_var = np.std([x.cpu().item() if isinstance(x, torch.Tensor) else x for x in precision_list])

    # Print results with variance
    print('Final Average Detection via ENT:')
    print('Test Accuracy: Mean={}, std={}'.format(test_acc_mean, test_acc_var))
    print('AUROC: Mean={}, std={}'.format(auroc_mean, auroc_var))
    print('AUPR_0: Mean={}, std={}'.format(aupr_0_mean, aupr_0_var))
    print('FPR95: Mean={}, std={}'.format(fprs_mean, fprs_var))
    print('Precision: Mean={}, std={}'.format(precision_mean, precision_var))
