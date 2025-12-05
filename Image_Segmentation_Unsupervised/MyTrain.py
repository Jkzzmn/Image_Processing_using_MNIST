import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import numpy as np

class MyTrain():
    def __init__(self, model,epochs):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(),lr=0.0001,weight_decay=1e-5)
        self.epochs = epochs
        self.epsilon = 1e-8
        self.reg_weight=1e-6

    def get_model(self):
        return self.model
    
    def train(self, dataset=None):
        self.model.train()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=10,       
        )

        for epoch in range(self.epochs):
            running_loss = 0.0
            for data,_ in dataset:
                self.optimizer.zero_grad()

                data_batch = data.unsqueeze(0)

                pred = self.model(data_batch)
                
                inside = pred * data_batch
                inside_sum = torch.sum(pred, dim=[2, 3], keepdim=True)
                alpha = torch.sum(inside, dim=[2, 3], keepdim=True) / (inside_sum + self.epsilon)

                outside = (1.0 - pred) * data_batch
                outside_sum = torch.sum(outside, dim=[2, 3], keepdim=True)
                beta = torch.sum(outside, dim=[2, 3], keepdim=True) / (outside_sum + self.epsilon)

                inside_loss = pred * torch.pow(data - alpha.detach(), 2)
                outside_loss = (1-pred) * torch.pow(data - beta.detach(), 2)

                loss_mean = torch.mean(inside_loss + outside_loss)

                dx = torch.pow(pred[:, :, :, :-1] - pred[:, :, :, 1:], 2)
                dy = torch.pow(pred[:, :, :-1, :] - pred[:, :, 1:, :], 2)

                loss_reg = torch.mean(dx) + torch.mean(dy)

                total_loss = loss_mean+ self.reg_weight * loss_reg

                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item() * data.size(0)

            epoch_loss = running_loss / len(dataset)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.6f}")
            scheduler.step(epoch_loss)