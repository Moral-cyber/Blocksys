# Module of Distributed Learning Functions
# Used for ResNet-18 Implementation

import torch, torchvision
from torchvision import datasets
from torchvision import transforms as trns
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
import time
import copy

def DataDistributer(train_dataset, K, cls = 0, het_th = 5):
    '''
    Distribute data among the clients
    train_dataset: the global dataset of any format (should have method len())
    K: number of clients
    cls: distribution class
        cls = 0 : randomly distributed among client. The sizes of local clients are also random
    het_th: minimum ratio of the largest dataset to the smallest
    It returns a sampler that can be given to the DataLoader
    '''
    # set the classes
    if cls == 0:
        h_0 = 1
        while h_0 < het_th:
            het = np.random.uniform(0,1,K)
            het /= np.sum(het)
            h_0 = np.max(het) / np.min(het)
    elif cls == 'debug':
        het = (60/len(train_dataset))*np.ones(K)
    else:
        raise ValueError('Invalid cls')
    
    # shuffel the indices
    idx = [i for i in range(len(train_dataset))]
    np.random.shuffle(idx)

    # make the sampler
    smplr = []
    for k in range(K-1):
        l = int(het[k] * len(train_dataset))
        smplr.append(idx[:l])
        idx = idx[l:]
    
    smplr.append(idx)
    return smplr, het

def FLtrainLoader(train_dataset, batch_size, client_sampler, K, num_workers = 0):
    '''
    Load data in a distributed fashion
    '''
    trainLoader = []
    trainLoader_itr = []
    for k in range(K):
        loader = DataLoader(train_dataset, batch_size=batch_size, sampler=client_sampler[k],num_workers=num_workers)
        trainLoader.append(loader)
        trainLoader_itr.append(loader._get_iterator())
    
    return trainLoader, trainLoader_itr



def ResNet18_cifar(device):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(in_features=512, out_features=10)
    model = model.to(device)
    return model

def FullModelCopy(model_dest, model_source):
    '''
    <model_source> is copied to <model_dest>
    all the parameters and gradients of <model_dist> are set to the ones in <mode_source>
    '''
    for param_1, param_2 in zip(model_dest.parameters(), model_source.parameters()):
        param_1.data = copy.deepcopy(param_2.data)
        param_1.grad = copy.deepcopy(param_2.grad)

def ParamLoader(model, params):
    '''
    <params> is copied to <model>
    all the parameters and gradients of <params> are copied in <model>
    '''
    for param_1, param_2 in zip(model.parameters(), params):
        param_1.data = copy.deepcopy(param_2.data)
        param_1.grad = copy.deepcopy(param_2.grad)

def totalSizeReader(model):
    '''
    Read the total number of model parameters
    '''
    model_size = 0
    for param in model.parameters():
        model_size += param.data.flatten().shape[0]

    return model_size

def TopK(model, numK, device, GSerror = None):
    '''
    Sparsifies <model> with top K algorithm 
    <numK> is the value of K in topK
    <GSerror> is the gradient sparsification error
              set it to <None> for the first round

    returns the sparsified model and the new GS error 
    '''
    g_aug = torch.tensor([])
    g_aug = g_aug.to(device)

    for param in model.parameters():
        #  check if there is any gradients in the model
        if param.grad == None:
            raise ValueError('The gradient is None!')
        

        # stack the models into a single vector
        grd_n = param.grad.flatten()
        g_aug = torch.cat([g_aug, grd_n])
    
    # set the GS error to zero for the fist round
    if GSerror == None:
        GSerror = torch.zeros(g_aug.shape)
        GSerror = GSerror.to(device)
    
    # make the accumulated gradient
    accum_grd = g_aug + GSerror

    # sparsify by selecting the top K
    grd_sprs = copy.deepcopy(accum_grd)
    accumSZ = torch.abs(accum_grd)
    _, idx_top = torch.sort(accumSZ, descending=True)
    grd_sprs[idx_top[numK:]] = 0

    # Update GS error
    Er_out = accum_grd - grd_sprs
    
    return grd_sprs, Er_out


def RegTopK(model, Del, numK, device, GSerror = None):
    '''
    Sparsifies <model> with RegTopK algorithm 
    <numK> is the value of K in topK
    <GSerror> is the gradient sparsification error
              set it to <None> for the first round

    returns the sparsified model and the new GS error 
    '''
    Mu = 5
    # vector of local gradients
    grd_loc = torch.tensor([]).to(device)

    # save the shape of each layer
    for param in model.parameters():
        grd_n = param.grad.detach().flatten()
        grd_loc = torch.cat([grd_loc, grd_n])
    
    
    # set the GS error to zero if not selected
    if GSerror == None:
        GSerror = torch.zeros(grd_loc.shape).to(device)
        
    # calculate local accumulated gradient
    a_loc = grd_loc + GSerror

    post = torch.abs(torch.tanh((Del+1)/Mu)) * torch.abs(a_loc)
    

    # choose K largest entries
    _, idx = torch.sort(post, descending=True)
    a_sprs = copy.deepcopy(a_loc)
    a_sprs[idx[numK:]] = 0
    sel = torch.zeros(a_loc.shape).to(device)
    sel[idx[:numK]] = 1


    # Update the GS Error
    Er_out = a_loc - a_sprs

    return a_sprs, Er_out, sel


def Model_extractor(model, device):

    mod_param = torch.tensor([]).to(device)
    for param in model.parameters():
        x_n = param.data.detach().flatten()
        mod_param = torch.cat([mod_param, x_n])
    
    return mod_param
