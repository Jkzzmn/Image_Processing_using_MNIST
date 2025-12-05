import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyTrain():
    def __init__(self,model,dataloader,epochs):
        self.model = model
        self.loss_cal = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(),lr=0.0001,weight_decay=1e-5)
        self.dataloader = dataloader
        self.epochs = epochs

    def train(self):
        self.model.train()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=10,       
        )

        for epoch in range(self.epochs):
            running_loss = 0.0
            for step, (data,mask) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                pred = self.model(data)

                loss = self.loss_cal(pred,mask)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * data.size(0)

            epoch_loss = running_loss / len(self.dataloader)
            scheduler.step(epoch_loss)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.6f}")



                






    