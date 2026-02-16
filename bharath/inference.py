import torch
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Initialize model
G = UNetGenerator().to(device)

# Load checkpoint
checkpoint = torch.load("/home/bharath/dl/pix2pix_checkpoints/exp2_lambda5/pix2pix_epoch_100.pth", map_location=device)
G.load_state_dict(checkpoint['generator_state_dict'])

G.eval()  
transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),                 # [0, 1]
                            transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))   # [-1, 1]
                        ])

def run_validation(val_dir, output_dir="./val_outputs/lambda5_epoch100/"):
    os.makedirs(output_dir, exist_ok=True)

    G.eval()
    count = 0
    for file in os.listdir(val_dir):
        img_path = os.path.join(val_dir, file)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Split image
        sketch = img.crop((0, 0, w//2, h))
        real_photo = img.crop((w//2, 0, w, h))

        # Transform sketch
        input_tensor = transform(sketch).unsqueeze(0).to(device)

        with torch.no_grad():
            fake_photo = G(input_tensor)

        # Convert from [-1,1] → [0,1]
        fake_photo = (fake_photo + 1) / 2

        # Convert real & sketch for visualization
        sketch_tensor = transform(sketch).unsqueeze(0)
        real_tensor = transform(real_photo).unsqueeze(0)

        sketch_tensor = (sketch_tensor + 1) / 2
        real_tensor = (real_tensor + 1) / 2

        # Concatenate side-by-side: sketch | generated | real
        comparison = torch.cat([sketch_tensor, fake_photo.cpu(), real_tensor], dim=3)

        save_image(comparison, os.path.join(output_dir, file))
        count += 1
        if count>=10:
            break
        # plt.figure(figsize=(12,4))
        # plt.subplot(1,3,1)
        # plt.title("Sketch")
        # plt.imshow(sketch_tensor.squeeze(0).permute(1,2,0))
        # plt.axis("off")

        # plt.subplot(1,3,2)
        # plt.title("Real Photo")
        # plt.imshow(real_tensor.squeeze(0).permute(1,2,0))
        # plt.axis("off")
        # plt.show()

        # plt.subplot(1,3,3)
        # plt.title("Output Photo")
        # plt.imshow(fake_photo.squeeze(0).cpu().permute(1,2,0))
        # plt.axis("off")
        # plt.show()
        

    print("Validation inference completed!")
path = "/media/Data_2/bharath_data_new/"
run_validation(path +"/val")
