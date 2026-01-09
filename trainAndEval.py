from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassRecall, MulticlassAccuracy, MulticlassPrecision
from torchmetrics import ConfusionMatrix
import torch
import numpy as np


def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0  
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Training', unit='batch')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)       
        optimizer.zero_grad()  
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)      
        preds = torch.argmax(outputs, dim=1)     
        accuracy_metric(preds, labels)
        iou_metric(preds, labels)
        precision_metric(preds, labels)
        f1_metric(preds, labels)
        confmat(preds, labels)  
        pbar.set_postfix({
            'Batch Loss': f'{loss.item():.4f}',
            'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
            'Mean IoU': f'{iou_metric.compute():.4f}',
            'Mean Precision': f'{precision_metric.compute():.4f}',
            'Mean F1 Score': f'{f1_metric.compute():.4f}'
        }) 
    epoch_loss = running_loss / len(dataloader.dataset)  
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()

    cm = confmat.compute().cpu().numpy() 
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    confmat.reset()
   
    return cm_normalized, epoch_loss, mean_accuracy, mean_iou, mean_precision, mean_f1

def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0    
    accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes).to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
    pbar = tqdm(dataloader, desc='Evaluating', unit='batch')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
                
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            # Update metrics
            accuracy_metric(preds, labels)
            iou_metric(preds, labels)
            precision_metric(preds, labels)
            f1_metric(preds, labels)
            confmat(preds, labels)  # confusion matrix
            # Update tqdm description with metrics
            pbar.set_postfix({
                'Batch Loss': f'{loss.item():.4f}',
                'Mean Accuracy': f'{accuracy_metric.compute():.4f}',
                'Mean IoU': f'{iou_metric.compute():.4f}',
                'Mean Precision': f'{precision_metric.compute():.4f}',
                'Mean F1 Score': f'{f1_metric.compute():.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_accuracy = accuracy_metric.compute().cpu().numpy()
    mean_iou = iou_metric.compute().cpu().numpy()
    mean_precision = precision_metric.compute().cpu().numpy()
    mean_f1 = f1_metric.compute().cpu().numpy()
    cm = confmat.compute().cpu().numpy() 
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 
    confmat.reset()
    model.float()
    return cm_normalized, epoch_loss, mean_accuracy, mean_iou, mean_precision, mean_f1