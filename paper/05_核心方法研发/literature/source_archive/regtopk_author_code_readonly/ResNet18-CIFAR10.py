# ResNet-18 over CIFAR-10 Implementation
# Consistent with what presented in Section 2 of the supplementary file

import torch, torchvision
from torchvision import datasets
from torchvision import transforms as trns
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
import time
import DistLearn as dl
import copy
import pandas as pd

# Device
device = torch.device("mps" if torch.has_mps else "cpu")


# Hyperparameters
lr = 0.01
batch_size = 20
num_rounds = 3000
test_dur = 10 # number of rounds after which we run a test

# FL Setting
Num_clients = 8
dist_cls = 0 # class used for dataset distribution
num_local_itr = [40 for _ in range(Num_clients)] # number of local batches
lr_loc = 0.01*np.ones(Num_clients)
sprs_factor = 0.0001

# Loading data
transform = trns.Compose([
    trns.Resize((224, 224)),
    trns.ToTensor(),
    trns.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset_train = datasets.CIFAR10("data", train=True, transform=transform, download=True)
dataset_test = datasets.CIFAR10("data", train=False, transform=transform, download=True)

# Load Data Distributed
smpler, het = dl.DataDistributer(dataset_train, Num_clients, cls=dist_cls, het_th=5)
trainLoader, trainLoader_itr = dl.FLtrainLoader(dataset_train, batch_size=batch_size, K=Num_clients, client_sampler=smpler)
testLoader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# Define Model
model_gl_topK = dl.ResNet18_cifar(device)
model_gl = dl.ResNet18_cifar(device)
dist_0 = float('inf')

# keep the shape of layer tensors in ResNet18
shape_memory = []
for param in model_gl.parameters():
        # read the length of the layer
        layer = param.data.detach().flatten()
        # save it in the memory
        shape_memory.append(layer.shape[0])


## Determine Sparsity and initiate RegTopK
model_size = dl.totalSizeReader(model_gl)
# determine the number f non-zero entries
num_K = int(sprs_factor * model_size)

## Initiate Sparsification Parameters
# accumulated gradients
a_gl = torch.zeros(model_size).to(device)
a_loc = []
a_loc_TopK = []

# GS error vectors
GS_error = []
GS_error_TopK = []


# List of clients selections
sel = []

# initite for each client
for k in range(Num_clients):
    a_loc.append(torch.zeros(model_size).to(device))
    a_loc_TopK.append(torch.zeros(model_size).to(device))

    GS_error.append(torch.zeros(model_size).to(device))
    GS_error_TopK.append(torch.zeros(model_size).to(device))

    sel.append(torch.zeros(model_size).to(device))


# Optimizer
Optimizer = torch.optim.SGD(model_gl.parameters(), lr=lr)
Optimizer_topK = torch.optim.SGD(model_gl_topK.parameters(), lr=lr)

# Start Training
start = time.time()

# Initiate the accuracy vector
num_tests = int(num_rounds/test_dur)
Accuracy = np.zeros(num_tests)
Accuracy_topK = np.zeros(num_tests)
ind_test = 0

# initiate the batch counter
batch_cnt = np.zeros(Num_clients)

    
for round in range(num_rounds):
    print(f"Starting communication round {round+1}")

    for k in range(Num_clients):
        # Start with making a local model 
        model_loc = dl.ResNet18_cifar(device)
        dl.FullModelCopy(model_loc, model_gl)

        model_loc_topK = dl.ResNet18_cifar(device)
        dl.FullModelCopy(model_loc_topK, model_gl_topK)

        Optimizer_loc = torch.optim.SGD(model_loc.parameters(), lr = lr_loc[k])
        Optimizer_loc_topK = torch.optim.SGD(model_loc_topK.parameters(), lr = lr_loc[k])
        
        for batch in range(num_local_itr[k]):
            Optimizer_loc.zero_grad()
            Optimizer_loc_topK.zero_grad()
            # assure that we are not going beyond the length of the iterator
            if batch_cnt[k] >= len(trainLoader_itr[k]):
                batch_cnt[k] = 0
                trainLoader_itr[k] = trainLoader[k]._get_iterator()
            
            
            # count the number batches past
            batch_cnt[k] += 1
            

            x_train, y_train = trainLoader_itr[k]._next_data()
            x_train = x_train.to(device)

            # get the soft info
            raw_pred = model_loc(x_train)
            raw_pred_topK = model_loc_topK(x_train)

            raw_pred = raw_pred.cpu()
            raw_pred_topK = raw_pred_topK.cpu()

            dist_pred = torch.softmax(raw_pred, dim=1)
            dist_pred_topK = torch.softmax(raw_pred_topK, dim=1)

            # calculate loss
            loss = Fn.cross_entropy(dist_pred,y_train)
            loss_topK = Fn.cross_entropy(dist_pred_topK,y_train)

            # SGD step
            loss.backward()
            Optimizer_loc.step()

            loss_topK.backward()
            Optimizer_loc_topK.step()

        # Determine Delta Factor for RegTopK
        a_not_k = Num_clients*(a_gl * sel[k]) + (1-sel[k])*(1e8) - a_loc[k]
        a_k_bar = a_loc[k] + (1-sel[k])
        Del_k = a_not_k / a_k_bar

        # Sparsify by RegTopK
        a_loc[k], GS_error[k], sel[k] = dl.RegTopK(model_loc, Del_k, num_K, device, GSerror = GS_error[k])

        # Sparsify by TopK
        a_loc_TopK[k], GS_error_TopK[k] = dl.TopK(model=model_loc_topK, numK = num_K, device=device, GSerror = GS_error_TopK[k])

    # Aggregate RegTopK local gradients
    a_gl = sum(a_loc)/Num_clients
    a_gl_loop = copy.deepcopy(a_gl)

    Optimizer.zero_grad()

    # Put <a_gl> back in the global model
    cnt = 0
    for param in model_gl.parameters():
        a_gl_n = a_gl_loop[:shape_memory[cnt]]
        a_gl_loop = a_gl_loop[shape_memory[cnt]:]
        a_Tensor_n = a_gl_n.reshape(param.data.shape)

        param.grad = copy.deepcopy(a_Tensor_n)
        cnt += 1


    Optimizer.step()
    # model_gl, dist_0 = dl.termination(model_gl, x_0, model_0, dist_0, device)

    # Aggregate TopK
    a_gl_topK = sum(a_loc_TopK)/Num_clients

    Optimizer_topK.zero_grad()
    
    # Put <a_gl_topK> back in the global model of TopK
    cnt = 0
    for param in model_gl_topK.parameters():
        a_gl_n_topK = a_gl_topK[:shape_memory[cnt]]
        a_gl_topK = a_gl_topK[shape_memory[cnt]:]
        a_Tensor_n_topK = a_gl_n_topK.reshape(param.data.shape)

        param.grad = copy.deepcopy(a_Tensor_n_topK)
        cnt += 1

    Optimizer_topK.step()

    # validate the model every <test_dur> iterations
    if (round+1) % test_dur == 0:
        for x_test, y_test in testLoader:
            # flg += 1
            with torch.no_grad():
                x_test = x_test.to(device)

                raw_lr = model_gl(x_test)
                raw_lr_topK = model_gl_topK(x_test)

                raw_lr = raw_lr.cpu()
                raw_lr_topK = raw_lr_topK.cpu()

                dist_lr = Fn.softmax(raw_lr, dim=1)
                y_lr = torch.argmax(dist_lr, dim=1)

                dist_lr_topK = Fn.softmax(raw_lr_topK, dim=1)
                y_lr_topK = torch.argmax(dist_lr_topK, dim=1)

                Accuracy[ind_test] += torch.sum(y_lr == y_test) / batch_size

                Accuracy_topK[ind_test] += torch.sum(y_lr_topK == y_test) / batch_size
        
        Accuracy[ind_test] /= (len(testLoader)/100)
        Accuracy_topK[ind_test] /= (len(testLoader)/100)
        print(f'Round {round+1} finished -- Accuracy of RegTopK is {Accuracy[ind_test]}')
        print(f'Round {round+1} finished -- Accuracy of TopK is {Accuracy_topK[ind_test]}')
        ind_test += 1

end = time.time()

print(f'Total Time for {num_rounds} rounds with Batches of Length {batch_size} and local updates of {num_local_itr} Batches = {end-start}')

# Plot accuracy
plt.plot([i+1 for i in range(num_tests)], Accuracy, label="RegTopK")
plt.plot([i+1 for i in range(num_tests)], Accuracy_topK, label="TopK")
plt.legend()
plt.show()

# Save Data
Data = [f"{i} {Accuracy_topK[i]}" for i in range(num_tests)]
Data.append("TopK Ends")
for i in range(num_tests):
    Data.append(f"{i} {Accuracy[i]}")
Data.append("RegTopK Ends")

# Save as a LaTex table
df = pd.DataFrame(Data)
df.style.hide(axis="index").to_latex('./Data_RegTopK.tex',hrules=True)