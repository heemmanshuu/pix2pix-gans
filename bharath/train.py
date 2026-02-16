

import torch
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/pix2pix_experiment/exp2_lambda5")


class Edges2ShoesDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: path to edges2shoes/train or edges2shoes/test
        """
        self.root_dir = root_dir
        self.image_files = sorted(os.listdir(root_dir))
        self.transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),                 # [0, 1]
                            transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))   # [-1, 1]
                        ])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        w, h = img.size
        w_half = w // 2

        # Split image into sketch and photo
        sketch = img.crop((0, 0, w_half, h))
        photo  = img.crop((w_half, 0, w, h))


        sketch = self.transform(sketch)
        photo = self.transform(photo)

        return sketch, photo

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("balraj98/edges2shoes-dataset")

# print("Path to dataset files:", path)

path = "/media/Data_2/bharath_data_new/"
# for item in os.listdir(path):
#   print(item)

train_dataset = Edges2ShoesDataset(
    root_dir=path +"/train"
)

test_dataset = Edges2ShoesDataset(
    root_dir=path +"/val"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# sketch, photo = next(iter(train_loader))
# print(sketch.shape, photo.shape)

import matplotlib.pyplot as plt

def denormalize(img):
    return img * 0.5 + 0.5

# sketch, photo = next(iter(train_loader))

# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.title("Sketch")
# plt.imshow(denormalize(sketch[0]).permute(1,2,0))
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.title("Photo")
# plt.imshow(denormalize(photo[0]).permute(1,2,0))
# plt.axis("off")
# plt.show()

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        x = torch.cat((x, skip), dim=1)
        return x

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # -------- Encoder --------
        self.down1 = DownBlock(in_channels, 64, normalize=False)   # 128 → 64
        self.down2 = DownBlock(64, 128)                            # 64 → 32
        self.down3 = DownBlock(128, 256)                           # 32 → 16
        self.down4 = DownBlock(256, 512)                           # 16 → 8
        self.down5 = DownBlock(512, 512)                           # 8 → 4
        self.down6 = DownBlock(512, 512)                           # 4 → 2
        self.down7 = DownBlock(512, 512)                           # 2 → 1

        # -------- Bottleneck --------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # -------- Decoder --------
        self.up1 = UpBlock(512, 512, dropout=True)
        self.up2 = UpBlock(1024, 512, dropout=True)
        self.up3 = UpBlock(1024, 512, dropout=True)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)

        # -------- Output Layer --------
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # Bottleneck
        bottleneck = self.bottleneck(d7)

        # Decoder + skip connections
        u1 = self.up1(bottleneck, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        """
        in_channels: 6 because we concatenate sketch (3) + photo (3)
        """
        super().__init__()

        def conv_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(in_channels, 64, normalize=False),   # 128 -> 64
            *conv_block(64, 128),                            # 64 -> 32
            *conv_block(128, 256),                           # 32 -> 16
            *conv_block(256, 512, stride=1),                # 16 -> 15
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # 15 -> 14
        )

    def forward(self, sketch, photo):
        # Concatenate input and target along channel dimension
        x = torch.cat([sketch, photo], dim=1)
        return self.model(x)

adversarial_loss = nn.BCEWithLogitsLoss()  # for stable GAN training
l1_loss = nn.L1Loss()

def discriminator_loss(D, sketch, real_photo, fake_photo):
    """
    D: PatchGAN discriminator
    sketch: input sketches
    real_photo: ground truth
    fake_photo: generated by generator
    """
    # Real pairs
    pred_real = D(sketch, real_photo)
    loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))

    # Fake pairs (detach to avoid backprop to generator)
    pred_fake = D(sketch, fake_photo.detach())
    loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))

    # Total loss
    loss_D = (loss_real + loss_fake) * 0.5
    return loss_D

def generator_loss(D, sketch, fake_photo, real_photo, lambda_l1=10):
    # Adversarial loss (want D to think fake is real)
    pred_fake = D(sketch, fake_photo)
    loss_GAN = adversarial_loss(pred_fake, torch.ones_like(pred_fake))

    # L1 reconstruction loss
    loss_L1 = l1_loss(fake_photo, real_photo)

    # Total generator loss
    loss_G = loss_GAN + lambda_l1 * loss_L1
    return loss_G, loss_GAN, loss_L1

# -----------------------------
# 4️⃣ Losses & Optimizers
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

optimizer_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

print(device)

# from google.colab import drive
# drive.mount('/content/drive')

save_path = "/home/bharath/dl/pix2pix_checkpoints/exp2_lambda5/"
os.makedirs(save_path, exist_ok=True)

# -----------------------------
# 5️⃣ Training Loop
# -----------------------------
num_epochs = 100
lambda_l1 = 5

# G_losses = []
# D_losses = []
# L1_losses = []
# GAN_losses = []

for epoch in range(num_epochs):
    epoch_G_loss = 0
    epoch_D_loss = 0
    epoch_GAN_loss = 0
    epoch_L1 = 0
    for i, (sketch, photo) in enumerate(train_loader):
        sketch = sketch.to(device)
        photo = photo.to(device)

        # -------------------
        #  Train Discriminator
        # -------------------
        optimizer_D.zero_grad()
        fake_photo = G(sketch)

        pred_real = D(sketch, photo)
        pred_fake = D(sketch, fake_photo.detach())

        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # -------------------
        #  Train Generator
        # -------------------
        optimizer_G.zero_grad()
        pred_fake = D(sketch, fake_photo)
        loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = criterion_L1(fake_photo, photo)
        loss_G = loss_G_GAN + lambda_l1 * loss_G_L1
        loss_G.backward()
        optimizer_G.step()

        epoch_G_loss += loss_G.item()
        epoch_D_loss += loss_D.item()
        epoch_GAN_loss += loss_G_GAN.item()
        epoch_L1 += loss_G_L1.item()

        

        # -------------------
        # Logging
        # -------------------
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(train_loader)}] "
                  f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, L1: {loss_G_L1.item():.4f}")

    epoch_G_loss /= len(train_loader)
    epoch_D_loss /= len(train_loader)
    epoch_GAN_loss /= len(train_loader)
    epoch_L1 /= len(train_loader)

    # G_losses.append(epoch_G_loss)
    # D_losses.append(epoch_D_loss)
    # GAN_losses.append(epoch_GAN_loss)
    # L1_losses.append(epoch_L1)

    torch.save({
    'epoch': epoch + 1,
    'generator_state_dict': G.state_dict(),
    'discriminator_state_dict': D.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, f"{save_path}/pix2pix_epoch_{epoch+1}.pth")
    print(f"Model saved at epoch {epoch+1}")

    writer.add_scalar("Loss/Generator", epoch_G_loss, epoch)
    writer.add_scalar("Loss/Discriminator", epoch_D_loss, epoch)
    writer.add_scalar("Loss/L1", epoch_L1, epoch)
    writer.add_scalar("Loss/GAN", epoch_GAN_loss, epoch)

# import torch
# from torchvision import transforms
# from PIL import Image
# import os

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Initialize model
# G = UNetGenerator().to(device)

# # Load checkpoint
# checkpoint = torch.load(f"{save_path}/pix2pix_epoch_{2}.pth", map_location=device)
# G.load_state_dict(checkpoint['generator_state_dict'])

# G.eval()   # VERY IMPORTANT

# transform = transforms.Compose([
#                             transforms.Resize((256, 256)),
#                             transforms.ToTensor(),                 # [0, 1]
#                             transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))   # [-1, 1]
#                         ])

# def run_validation(val_dir, output_dir="val_outputs"):
#     os.makedirs(output_dir, exist_ok=True)

#     G.eval()

#     for file in os.listdir(val_dir):
#         img_path = os.path.join(val_dir, file)

#         img = Image.open(img_path).convert("RGB")
#         w, h = img.size

#         # Split image
#         sketch = img.crop((0, 0, w//2, h))
#         real_photo = img.crop((w//2, 0, w, h))

#         # Transform sketch
#         input_tensor = transform(sketch).unsqueeze(0).to(device)

#         with torch.no_grad():
#             fake_photo = G(input_tensor)

#         # Convert from [-1,1] → [0,1]
#         fake_photo = (fake_photo + 1) / 2

#         # Convert real & sketch for visualization
#         sketch_tensor = transform(sketch).unsqueeze(0)
#         real_tensor = transform(real_photo).unsqueeze(0)

#         sketch_tensor = (sketch_tensor + 1) / 2
#         real_tensor = (real_tensor + 1) / 2

#         # Concatenate side-by-side: sketch | generated | real
#         comparison = torch.cat([sketch_tensor, fake_photo.cpu(), real_tensor], dim=3)

#         #save_image(comparison, os.path.join(output_dir, file))
#         plt.figure(figsize=(12,4))
#         plt.subplot(1,3,1)
#         plt.title("Sketch")
#         plt.imshow(sketch_tensor.squeeze(0).permute(1,2,0))
#         plt.axis("off")

#         plt.subplot(1,3,2)
#         plt.title("Real Photo")
#         plt.imshow(real_tensor.squeeze(0).permute(1,2,0))
#         plt.axis("off")
#         plt.show()

#         plt.subplot(1,3,3)
#         plt.title("Output Photo")
#         plt.imshow(fake_photo.squeeze(0).cpu().permute(1,2,0))
#         plt.axis("off")
#         plt.show()
#         break

#     print("Validation inference completed!")

# run_validation(path +"/val")

