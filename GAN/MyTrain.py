import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
       nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
       nn.init.normal_(m.weight.data, 1.0, 0.02)
       nn.init.constant_(m.bias.data, 0)

class MyTrain():
    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        self.generator      = generator
        self.discriminator  = discriminator
        self.generator.train()
        self.discriminator.train()
        self.loss_generator     = [] # generator loss per epoch (average over batches)
        self.loss_discriminator = [] # discriminator loss per epoch (average over batches)   
        
        self.criterion = nn.BCELoss()

        #self.optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))
        #self.optimizer_G = optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999))

        self.optimizer_D = optim.AdamW(
            discriminator.parameters(),
            lr=0.00005,
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )
        self.optimizer_G = optim.AdamW(
            generator.parameters(),
            lr=0.0001,
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )


        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        self.batch_size = 100
        self.dim_latent = 100

    def get_generator(self):
        return self.generator


    def get_discriminator(self):
        return self.discriminator


    def get_loss_generator(self):
        return self.loss_generator
    
      
    def get_loss_discriminator(self):
        return self.loss_discriminator
    
        
    def train(self, dataset: Dataset):
        # train models and save losses
        real_label = 0.8
        fake_label = 0.

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        num_epoch = 20000                                              # dummy implementation      
        for epoch in range(num_epoch):                                      # dummy implementation
            total_g_loss = 0.0
            total_d_loss = 0.0

            for i, data in enumerate(dataloader, 0):
                real_images = data[0]
                if real_images.dim() == 3:
                    real_images = real_images.unsqueeze(1)
                b_size = real_images.size(0)

                real_labels = torch.full((b_size,), real_label, dtype=torch.float)
                fake_labels = torch.full((b_size,), fake_label, dtype=torch.float)

                self.discriminator.zero_grad()

                output = self.discriminator(real_images).view(-1)
                loss_D_real = self.criterion(output, real_labels)
                loss_D_real.backward()

                noise = torch.randn(b_size, self.dim_latent, 1, 1)
                fake_images = self.generator(noise)

                output = self.discriminator(fake_images.detach()).view(-1) 
                loss_D_fake = self.criterion(output, fake_labels)
                loss_D_fake.backward()

                loss_D = loss_D_real + loss_D_fake
                self.optimizer_D.step()
                total_d_loss += loss_D.item()


                self.generator.zero_grad()

                output = self.discriminator(fake_images).view(-1)
                loss_G = self.criterion(output, real_labels)
                loss_G.backward()

                self.optimizer_G.step()

                total_g_loss += loss_G.item()

            avg_g_loss = total_g_loss / len(dataloader)
            avg_d_loss = total_d_loss / len(dataloader)
            self.loss_generator.append(avg_g_loss)
            self.loss_discriminator.append(avg_d_loss)
            

            print(f'Epoch [{epoch+1}/{num_epoch}] G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}')

            with torch.no_grad():
                noise = torch.randn(16, self.dim_latent, 1, 1)
                fake = self.generator(noise)
                print(f'Epoch {epoch+1} fake min/max:', fake.min().item(), fake.max().item())

