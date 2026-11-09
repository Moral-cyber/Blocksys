# This is the code for the toy example in the paper
# Logistic Regression with only 2 clients and J=2 model parameters

import torch
import numpy as np
import pandas
import matplotlib.pyplot as plt

# Define the model
class model():
    def __init__(self, w):
        self.weight = w
    
    def forward(self, x):
        z = np.dot(self.weight,x)
        return 1 / (1+np.exp(-z))
    
    def loss(self, X, Y):
        l = 0
        for x, y in zip(X,Y):
            p = self.forward(x)
            l += -y * np.log(p) - (1-y)*np.log(1-p)
        return l/len(X)
    
    def grad(self, X, Y):
        g = 0
        for x, y in zip(X,Y):
            p = self.forward(x)
            g += (p-y)*x
        return g/len(Y)
    
    def sgd(self, grd, eta):
        self.weight -= eta*grd
    
# Starting point
w = np.array([0, 1], dtype=np.float64)
model_c = model(w.copy())
model_topK = model(w.copy())
model_regK = model(w.copy())

# Data points
x_1, y_1 = np.array([100, 1], dtype=np.float64), float(1)
x_2, y_2 = np.array([-100, 1], dtype=np.float64), float(1)
X_c, Y_c = [x_1, x_2], [y_1, y_2]


err_1, err_2 = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
err_rg_1, err_rg_2 = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
Delta_1, Delta_2 = 1e8*np.ones(2, dtype=np.float64), 1e8*np.ones(2, dtype=np.float64)

# Training loop
iter = 150
eta = .9

loss_c = [model_c.loss(X_c, Y_c)]
loss_topK = [model_topK.loss(X_c, Y_c)]
loss_regK = [model_regK.loss(X_c, Y_c)]

for i in range(iter):
    # centralized
    grd_c = model_c.grad(X_c, Y_c)
    model_c.sgd(grd_c, eta)
    loss_c.append(model_c.loss(X_c, Y_c))

    # topK
    grd_1 = model_topK.grad([x_1], [y_1])
    grd_2 = model_topK.grad([x_2], [y_2])
    a_1 = grd_1 + err_1
    a_2 = grd_2 + err_2

    idx_1 = np.argmax(np.abs(a_1))
    idx_2 = np.argmax(np.abs(a_2))

    g_1, g_2 = np.zeros(grd_1.size), np.zeros(grd_2.size)
    g_1[idx_1], g_2[idx_2] = a_1[idx_1], a_2[idx_2]
    err_1, err_2 = a_1 - g_1, a_2 - g_2

    model_topK.sgd((g_1+g_2)/2, eta)
    loss_topK.append(model_topK.loss(X_c, Y_c))

    # regK
    grd_rg_1 = model_regK.grad([x_1], [y_1])
    grd_rg_2 = model_regK.grad([x_2], [y_2])
    a_rg_1 = grd_rg_1 + err_rg_1
    a_rg_2 = grd_rg_2 + err_rg_2

    metric_1, metric_2 = np.abs(np.tanh(1+Delta_1))*np.abs(a_rg_1), np.abs(np.tanh(1+Delta_2))*np.abs(a_rg_2)
    idx_rg_1, idx_rg_2 = np.argmax(metric_1), np.argmax(metric_2)
  

    g_rg_1, g_rg_2 = np.zeros(grd_rg_1.size), np.zeros(grd_rg_2.size)
    g_rg_1[idx_rg_1], g_rg_2[idx_rg_2] = a_rg_1[idx_rg_1], a_rg_2[idx_rg_2]

    err_rg_1, err_rg_2 = a_rg_1 - g_rg_1, a_rg_2 - g_rg_2
    Delta_1, Delta_2 = 1e8*np.ones(2, dtype=np.float64), 1e8*np.ones(2, dtype=np.float64)
    Delta_1[idx_rg_1], Delta_2[idx_rg_2] = g_rg_2[idx_rg_1]/g_rg_1[idx_rg_1], g_rg_1[idx_rg_2]/g_rg_2[idx_rg_2]


    model_regK.sgd((g_rg_1+g_rg_2)/2, eta)
    loss_regK.append(model_regK.loss(X_c, Y_c))

   

# Plot the loss against iterations
plt.plot([i for i in range(iter+1)], loss_c, label = 'Dist SGD')
plt.plot([i for i in range(iter+1)], loss_topK, label = 'TopK')
plt.plot([i for i in range(iter+1)], loss_regK, label = 'RegTopK')
plt.legend()
plt.show()

# Save Data
Data = [f"{i}    {loss_c[i]}" for i in range(iter+1)]
Data.append("Dist SGD Ends")
for i in range(iter+1):
    Data.append(f"{i}    {loss_topK[i]}")
Data.append("TopK Sparsification Ends")
for i in range(iter+1):
    Data.append(f"{i}    {loss_regK[i]}")
Data.append("RegTopK Sparsification Ends")

# Save as a LaTex Table
df = pandas.DataFrame(Data)
df.style.hide(axis="index").to_latex('./Data/Toy_Example.tex',hrules=True)