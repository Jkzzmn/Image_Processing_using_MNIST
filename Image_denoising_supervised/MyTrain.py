import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MyTrain():
    def __init__(self,model,dataloader,epochs,reg):
        self.model = model
        self.MSEloss = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(),lr=0.0001,weight_decay=1e-5)
        self.dataloader = dataloader
        self.epochs = epochs
        self.reg = reg
        
    def gradient_loss(self,output):
        grad_x = output[:, :, 1:, :] - output[:, :, :-1, :]
        grad_y = output[:, :, :, 1:] - output[:, :, :, :-1]
        return torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))

    
    def train(self,input):

        self.model.train()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=10,       
        )

        for epoch in range(self.epochs):
            running_loss = 0.0
            for step, (data,_) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                mask = (torch.rand_like(input) > 0.97).float()  # 3% masking
                masked_input = input * (1 - mask)

                output = self.model(masked_input)

                recon_loss = self.MSEloss(output, input)
                mask_loss = self.MSEloss(output * mask, input * mask)
                LOSS1 = 0.3 * mask_loss + 0.7 * recon_loss
                
                LOSS2 = self.gradient_loss(output)
                total_loss = LOSS1 + self.reg * LOSS2

                total_loss.backward()
                
                self.optimizer.step()

                running_loss += total_loss.item() * data.size(0)

            epoch_loss = running_loss / len(self.dataloader)
            scheduler.step(epoch_loss)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.6f}")


        
        

