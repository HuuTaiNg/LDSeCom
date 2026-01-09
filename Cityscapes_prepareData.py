import torch
from torchvision.datasets import Cityscapes
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

# Mapping from Cityscapes original IDs to train IDs (34 → 19 classes)
CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0,
    8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
    16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18
}

def convert_to_train_id(mask, ignore_to_zero=False):
    mask_np = np.array(mask)
    train_id_mask = np.full(mask_np.shape, 255, dtype=np.uint8)
    for id_, train_id in CITYSCAPES_ID_TO_TRAINID.items():
        train_id_mask[mask_np == id_] = train_id
    if ignore_to_zero:
        train_id_mask[train_id_mask == 255] = 0
    return torch.as_tensor(train_id_mask, dtype=torch.long)

class ImgTransform:
    def __init__(self, resize=None):
        self.resize = resize
        self.to_tensor = T.ToTensor()

    def __call__(self, img):
        if self.resize:
            img = TF.resize(img, self.resize, interpolation=Image.BILINEAR)
        return self.to_tensor(img)

class TargetTransform:
    def __init__(self, resize=None, ignore_to_zero=False):
        self.resize = resize
        self.ignore_to_zero = ignore_to_zero

    def __call__(self, target):
        if self.resize:
            target = TF.resize(target, self.resize, interpolation=Image.NEAREST)
        return convert_to_train_id(target, self.ignore_to_zero)

def get_cityscapes_loaders(root_path, resize=(256, 512), ignore_to_zero=False):
    transform = ImgTransform(resize)
    target_transform = TargetTransform(resize, ignore_to_zero)

    train_dataset = Cityscapes(
        root=root_path, split='train', mode='fine',
        target_type='semantic', transform=transform,
        target_transform=target_transform
    )
    val_dataset = Cityscapes(
        root=root_path, split='val', mode='fine',
        target_type='semantic', transform=transform,
        target_transform=target_transform
    )
    test_dataset = Cityscapes(
        root=root_path, split='test', mode='fine',
        target_type='semantic', transform=transform,
        target_transform=target_transform
    )

    return train_dataset, val_dataset, test_dataset