import torch
import torch.nn as nn
import torch.nn.functional as F


class MyOptim():
    def __init__(self,weight,lr = 0.01):
        self.weight = weight
        self.loss = None
        self.loss_batch = None
        self.grad = torch.zeros_like(self.weight)
        self.lr = lr

    def Cal_loss(self,pred,label):
        eps = 1e-7
        pred = torch.clamp(pred,eps,1-eps)
        loss = - (label * torch.log(pred) + (1 - label) * torch.log(1 - pred))
        self.loss = loss.mean()
        return self.loss

    
    def Cal_grad(self,pred,label,vec):
        eps = 1e-7
        pred = torch.clamp(pred,eps,1-eps)
        grad = torch.matmul(vec.T,(pred - label))/ vec.shape[0]
        self.grad = grad.reshape((1024,1))
    
    def step(self):
        with torch.no_grad():
            self.weight.data -= self.lr * self.grad

    def zero_grad(self):
        self.grad.zero_()
    


