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
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from checkData import show_sample
from fvcore.nn import FlopCountAnalysis

if __name__ == '__main__':
    # ---------------- Preparing dataset -------------------
    root = 'C:\\Users\\CCE1\\Downloads\\J05\\Cityscapes_dataset'
    train_dataset, val_dataset, test_dataset = get_cityscapes_loaders(root, resize=(512, 1024), ignore_to_zero=True)
    print(f"Number of train dataset: {len(train_dataset)}")
    print(f"Number of val dataset: {len(val_dataset)}")
    print(f"Number of test dataset: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    # ---------------- Preparing model ---------------------
    num_classes = 19
    segmentation_model = senderModel(num_classes=num_classes)

    def count_parameters(model):  
        return sum(p.numel() for p in model.parameters())
    total_params = count_parameters(segmentation_model)
    print(f"Total parameters: {total_params}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"Device: {device}")
    segmentation_model.to(device)
    segmentation_model = nn.DataParallel(segmentation_model)


    # ---------------- Training model ---------------------
    criterion = nn.CrossEntropyLoss()  
    optimizer = Adam(segmentation_model.parameters(), lr=1e-4)
    num_epochs = 800  
    epoch_saved = 0
    best_val_mAcc = 0.0  
    best_model_state = None
    segmentation_model.load_state_dict(torch.load('C:\\Users\\CCE1\\Downloads\\J05\\model_epoch.pth'))

    for epoch in range(num_epochs):
        cm_normalized_train, epoch_loss_train, mAcc_train, mIoU_train, mPre_train, mF1_train = train_epoch(segmentation_model, train_loader, criterion, optimizer, device, num_classes)
        cm_normalized_val, epoch_loss_val, mAcc_val, mIoU_val, mPre_val, mF1_val = evaluate(segmentation_model, val_loader, criterion, device, num_classes) 
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {mAcc_train:.4f}, Mean IoU: {mIoU_train:.4f}, Mean Precision: {mPre_train:.4f}, , Mean F1: {mF1_train:.4f}")
        print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}, Mean Precision: {mPre_val:.4f}, , Mean F1: {mF1_val:.4f}")
        f = open('training.txt', 'a')
        f.write(f"Epoch {epoch + 1}/{num_epochs}\n")
        f.write(f"Train Loss: {epoch_loss_train:.4f}, Mean Accuracy: {mAcc_train:.4f}, Mean IoU: {mIoU_train:.4f}, Mean Precision: {mPre_train:.4f}, Mean F1: {mF1_train:.4f}\n")
        f.write(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}, Mean Precision: {mPre_val:.4f}, , Mean F1: {mF1_val:.4f}\n")
        f.close()
        torch.save(segmentation_model.state_dict(), f"model_epoch{epoch}.pth")
        if mAcc_val >= best_val_mAcc:
            epoch_saved = epoch + 1 
            best_val_mAcc = mAcc_val
            best_model_state = copy.deepcopy(segmentation_model.state_dict())
            
    torch.save(best_model_state, "LSNet.pth")
    segmentation_model.load_state_dict(best_model_state)

    if isinstance(segmentation_model, torch.nn.DataParallel):
        segmentation_model = segmentation_model.module
    model_save = torch.jit.script(segmentation_model)

    model_save.save("LSNet.pt")
