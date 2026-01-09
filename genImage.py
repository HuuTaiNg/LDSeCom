from Cityscapes_prepareData import get_cityscapes_loaders
from trainAndEval import train_epoch, evaluate
from models import receiverModel
import torch.optim as optim
from torch.optim import Adam
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch
import copy
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import DDPMScheduler, DDPMPipeline
from fvcore.nn import FlopCountAnalysis

# ---------------- Preparing dataset -------------------
root = 'C:\\Users\\CCE1\\Downloads\\J05\\Cityscapes_dataset'
train_dataset, val_dataset, test_dataset = get_cityscapes_loaders(root, resize=(256, 512), ignore_to_zero=True)
print(f"Number of train dataset: {len(train_dataset)}")
print(f"Number of val dataset: {len(val_dataset)}")
print(f"Number of test dataset: {len(test_dataset)}")


train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

# -------------------------  Training Diffusion Model  ------------------------------------
AFMNet = receiverModel()
def count_parameters(model):  
    return sum(p.numel() for p in model.parameters())
total_params = count_parameters(AFMNet)
print(total_params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
AFMNet = AFMNet.to(device)
AFMNet = nn.DataParallel(AFMNet)

# -------------- Noise scheduler
total_step = 1000
noise_scheduler = DDPMScheduler(num_train_timesteps=total_step)
optimizer = Adam(AFMNet.parameters(), lr=2e-4)

epochs = 1000

AFMNet.train()
for epoch in range(epochs):
    pbar = tqdm(train_loader, desc='Training', unit='batch')
    for images, segmented_images in pbar:
        images = images.to(device)
        segmented_images = segmented_images.unsqueeze(1).to(device)
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps,
            (images.shape[0],), device=images.device
        ).long()
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
        noise_pred = AFMNet(noisy_images, segmented_images, timesteps, total_step)
        loss = nn.MSELoss()(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

torch.save(AFMNet.state_dict(), f"AFMNet_{epoch}_trained.pth")