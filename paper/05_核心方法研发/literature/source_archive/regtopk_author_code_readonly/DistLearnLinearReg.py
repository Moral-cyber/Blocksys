# Dist Learning Functions used in Linear Regression Implementation

import torch
import copy


def next_batch(cnt, train_set, batch_size):
    if cnt >= len(train_set[1]):
         idx = 0
    else:
         idx = cnt
            
    ind_0 = idx
    ind_1 = idx + batch_size
    
    if ind_1 <= len(train_set[1]):
        None
    else:
        ind_1 = len(train_set[1])
    
    cnt_new = cnt + batch_size
    X_train, y_train = train_set[0][ind_0:ind_1], train_set[1][ind_0:ind_1]
    return cnt_new, X_train, y_train

def topK(W: torch.Tensor, SprEr, num_k, device = 'cpu'):
    grd_topK = W.flatten()
    a_topK = grd_topK + SprEr

    idx_topK = torch.argsort(torch.abs(a_topK), descending=True)
    idx_topK = idx_topK[:num_k]
    
    grd_sprs_topK = torch.zeros(len(SprEr)).to(device)
    grd_sprs_topK[idx_topK] = a_topK[idx_topK]
    W_spr = grd_sprs_topK.reshape(W.shape)

    SprEr_upd = a_topK - grd_sprs_topK

    return W_spr, grd_sprs_topK, SprEr_upd 

def RegTopK(W: torch.Tensor, SprEr, Params, num_k, device = 'cpu'):
    Delta, Mu = Params

    grd_regK = W.flatten()
    a_regK = grd_regK + SprEr

    metric = torch.abs(torch.tanh((1+Delta)/Mu)) * torch.abs(a_regK) 

    idx_regK = torch.argsort(torch.abs(metric), descending=True)
    idx_regK = idx_regK[:num_k]
    
    grd_sprs_regK = torch.zeros(len(SprEr)).to(device)
    grd_sprs_regK[idx_regK] = a_regK[idx_regK]
    W_spr = grd_sprs_regK.reshape(W.shape)

    SprEr_upd = a_regK - grd_sprs_regK

    sel_regK = torch.zeros(len(SprEr)).to(device)
    sel_regK[idx_regK] = 1

    return W_spr, grd_sprs_regK, SprEr_upd , sel_regK

def model_copier(model: torch.nn.Linear):
    model_rep = copy.deepcopy(model)
    if model.weight.grad != None:
        model_rep.weight.grad = model.weight.grad.detach()
    elif model.bias != None:
        model_rep.bias.grad = model.bias.grad.detach()
    return model_rep

def termination(model: torch.nn.Linear, x_0: torch.Tensor, purt_0):
    x_new = copy.deepcopy(model.weight.data.detach())
    purt_th = torch.norm(x_0.flatten() - x_new.flatten(), p=2)**2

    if purt_th >= purt_0:
        # tau = (0.99)**(purt_th.item()/purt_0)
        # model.weight.data = copy.deepcopy((1-tau)*x_0.detach() + tau*x_new.detach())
        model.weight.data = copy.deepcopy(x_0.detach())
    else:
        purt_0 = 100*copy.deepcopy(purt_th.detach().item())
    return model, purt_0

def Delta_updater(grd_aggr, grd_clients, selection_clients):
    Delta_neu = []
    num_clnt = len(grd_clients)
    grd_sum_regK = grd_aggr * num_clnt
    for k in range(num_clnt):
        denum = grd_clients[k] + 1 - selection_clients[k]
        Delta_neu.append(selection_clients[k]*((grd_sum_regK - grd_clients[k])/denum) + (1e8) *(1 - selection_clients[k]))
    return Delta_neu