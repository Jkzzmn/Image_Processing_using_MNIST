import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from MyModel import MyVariationalEncoder
import torch.optim as optim


class MyTrain():
    def __init__(self, vae: MyVariationalEncoder):
        self.vae = vae
        self.vae.encoder.train()
        self.vae.decoder.train() 
        self.loss_epoch = [] # loss per epoch (average over batches)
        self.num_epoch  = 50000
        self.optimizer  = optim.AdamW(self.vae.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epoch)

    def cal_loss(self, x_recons,x,mu,logvar):
        recon_loss = F.l1_loss(x_recons, x, reduction='sum')

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        kl_weight = 1

        total_loss = (recon_loss + (kl_loss * kl_weight)) / x.size(0)

        return total_loss,recon_loss,kl_loss

    def train(self, dataset: Dataset):

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # train models and save losses
        for epoch in range(self.num_epoch):         # dummy implementation
            epoch_loss = 0

            for i, data in enumerate(dataloader):
                x = data[0]

                if x.dim() == 3: 
                    x = x.unsqueeze(1)

                if x.max() > 1.0:
                    x = x / 255.0

                x_recons, mu,logvar = self.vae(x)

                loss, recon, kl = self.cal_loss(x_recons,x,mu,logvar)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            self.scheduler.step()
            avg_loss = epoch_loss / len(dataloader)
            self.loss_epoch.append(avg_loss)


            print(f"Epoch [{epoch+1}/{self.num_epoch}] Loss: {avg_loss:.4f}")



