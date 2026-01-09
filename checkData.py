from Cityscapes_prepareData import get_cityscapes_loaders
from trainAndEval import train_epoch, evaluate
from models import senderModel
import torch.optim as optim
from torch.optim import Adam
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch
import copy
import segmentation_models_pytorch as smp
from cityscapesscripts.helpers.labels import labels 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def save_image_and_mask(img_np, mask_color, save_dir="outputs", name_prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)
    # If the image is float type with value range in [0, 1], it will convert to uint8 type
    if img_np.dtype != 'uint8':
        img_np = (img_np * 255).clip(0, 255).astype('uint8')

    Image.fromarray(img_np).save(os.path.join(save_dir, f"{name_prefix}_image.png"))
    Image.fromarray(mask_color).save(os.path.join(save_dir, f"{name_prefix}_mask.png"))

    print(f"✅ Saved to {save_dir}/{name_prefix}_image.png and _mask.png")

id2color = {
    0: [128, 64,128],   # road
    1: [244, 35,232],   # sidewalk
    2: [ 70, 70, 70],   # building
    3: [102,102,156],   # wall
    4: [190,153,153],   # fence
    5: [153,153,153],   # pole
    6: [250,170, 30],   # traffic light
    7: [220,220,  0],   # traffic sign
    8: [107,142, 35],   # vegetation
    9: [152,251,152],   # terrain
    10: [ 70,130,180],  # sky
    11: [220, 20, 60],  # person
    12: [255,  0,  0],  # rider
    13: [  0,  0,142],  # car
    14: [  0,  0, 70],  # truck
    15: [  0, 60,100],  # bus
    16: [  0, 80,100],  # train
    17: [  0,  0,230],  # motorcycle
    18: [119, 11, 32],  # bicycle
    255: [0, 0, 0]      # ignored
}

def label_to_color(mask):
    """
    Convert a 2D trainId mask (H, W) into a color RGB mask (H, W, 3).
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for train_id, color in id2color.items():
        color_mask[mask == train_id] = color
    return color_mask

def show_sample(image, mask):
    """
    Display an RGB image and its corresponding segmentation mask in color.
    """
    image_np = image.cpu().permute(1, 2, 0).numpy()  # [C, H, W] → [H, W, C]
    mask_np = mask.cpu().numpy()                     # [H, W]
    mask_color = label_to_color(mask_np)             # [H, W, 3]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(image_np)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(mask_color)
    axs[1].set_title("Segmentation Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
    # save_image_and_mask(image_np, mask_color, name_prefix="demo1")



# # ---------------- Preparing dataset -------------------
root = 'C:\\Users\\CCE1\\Downloads\\J05\\Cityscapes_dataset'
train_dataset, val_dataset, test_dataset = get_cityscapes_loaders(root, resize=(512,1024), ignore_to_zero=True)    
image, mask = val_dataset[10]
show_sample(image, mask)

from tqdm import tqdm

def get_unique_classes(dataset):
    unique_ids = set()
    for _, mask in tqdm(dataset, desc="Scanning dataset"):
        unique = torch.unique(mask)
        unique_ids.update(unique.tolist())
    return sorted(unique_ids)

unique_class_ids = get_unique_classes(val_dataset)
print("Class IDs:", unique_class_ids)
print("The number of class:", len(unique_class_ids))
   