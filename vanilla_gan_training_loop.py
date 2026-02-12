import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0002            # Learning Rate (Standard for GANs)
BATCH_SIZE = 64
Z_DIM = 100            # Latent vector size (Noise input)
IMG_DIM = 28 * 28      # 784 pixels
EPOCHS = 50
WARMUP_EPOCHS = 3      # Reduced warmup epochs for faster G convergence

# Custom Dataset for Fashion MNIST CSV
class FashionCSVDataset(Dataset):
    def __init__(self, csv_file):
        # Load CSV
        data = pd.read_csv(csv_file)
        # First column is label, the rest are pixels
        self.pixels = data.iloc[:, 1:].values.astype(np.float32)
        # Normalize from [0, 255] to [-1, 1]
        self.pixels = (self.pixels - 127.5) / 127.5
        
    def __len__(self):
        return len(self.pixels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.pixels[idx]), 0 # Returning dummy label 0

# Data Loading
dataset = FashionCSVDataset("fashion-mnist_train.csv")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(IMG_DIM, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(Z_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, IMG_DIM),
            nn.Tanh()  # Output range [-1, 1] to match Normalized Images
        )

    def forward(self, x):
        return self.gen(x)

# Initialize
disc = Discriminator().to(DEVICE)
gen = Generator().to(DEVICE)

# Optimizers
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))

# Loss Function (Includes Sigmoid internally for stability)
criterion = nn.BCEWithLogitsLoss()

# Schedulers
sched_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=30, gamma=0.1)
sched_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=30, gamma=0.1)

print("Starting Training Loop...")

for epoch in range(EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        
        real = real.to(DEVICE)
        batch_size = real.shape[0]
        
        # Labels for BCEWithLogitsLoss
        real_labels = torch.ones(batch_size, 1).to(DEVICE)
        fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

        # DISCRIMINATOR TRAINING 

        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake = gen(noise)

        # Discriminator Loss on Real Images
        disc_real = disc(real)
        loss_d_real = criterion(disc_real, real_labels)

        # Discriminator Loss on Fake Images
        disc_fake = disc(fake.detach())
        loss_d_fake = criterion(disc_fake, fake_labels)

        # Combine and Backprop
        loss_d = (loss_d_real + loss_d_fake) / 2
        disc.zero_grad()
        loss_d.backward()
        opt_disc.step()

        # GENERATOR TRAINING 
        
        # Warmup Check
        if epoch < WARMUP_EPOCHS:
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(loader)} "
                      f"Loss D: {loss_d:.4f} (Warmup)")
            continue

        # If not in warmup, train Generator
        output = disc(fake)
        loss_g = criterion(output, real_labels)

        gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(loader)} "
                  f"Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

    # Save images every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake = gen(noise).reshape(-1, 1, 28, 28)
            save_image(fake, f"samples/epoch_{epoch+1}.png", normalize=True)

    # Step the Schedulers
    sched_disc.step()
    sched_gen.step()
