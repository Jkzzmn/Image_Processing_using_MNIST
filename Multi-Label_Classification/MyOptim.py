import torch
import torch.nn as nn
import torch.nn.functional as F


class MyOptim():
    def __init__(self,model,lr = 0.01):
        self.model = model
        self.loss = None
        self.grad_W1 = None
        self.grad_W2 = None
        self.lr = lr

    def Cal_loss(self, pred, label):
        N = label.shape[0]
        loss = -torch.sum(label * torch.log(pred+1e-9)) / N
        self.loss = loss
        return loss

    def Cal_grad(self,x,a1,pred,label):
        N = label.shape[0]
        dL_dh2 = (pred - label) / N
        self.grad_W2 = a1.T @ dL_dh2
        dL_da1 = dL_dh2 @ self.model.weight2.T
        dL_dh1 = dL_da1 * (a1 * (1 - a1))
        self.grad_W1 = x.T @ dL_dh1



    def zero_grad(self):
        self.grad_W1 = torch.zeros_like(self.model.weight1)
        self.grad_W2 = torch.zeros_like(self.model.weight2)

    def step(self):
        with torch.no_grad():
            self.model.weight1 -= self.lr * self.grad_W1
            self.model.weight2 -= self.lr * self.grad_W2
            
    

