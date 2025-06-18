import os, yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
# 
from pipeline.utils import AverageMeter, iou_score
from pipeline.network import UNetPP
from pipeline.constants import *

def train(
    deep_sup: bool,
    train_loader : torch.utils.data.DataLoader,
    model : UNetPP, 
    criterion : nn.BCEWithLogitsLoss, 
    optimizer : optim.Adam,
    scaler: GradScaler) -> OrderedDict:
    """
    Training pipeline, along with mixed precision.
    
    :param deep_sup(bool):
    :param train_loader(torch.utils.data.DataLoader): Train Data Loader   
    :param model(UNetPP): UNet++ deined model   
    :param criterion:   
    :param optimizer:   
    :param scaler:   
    """
    
    
    # Metrices
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    
    model.train()
    
    # progress bar
    pbar = tqdm(total=len(train_loader))
    
    for input, target, _ in train_loader:
        
        #? 'cuda:0'
        
        input, target = input.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()
        with autocast(device_type=DEVICE_STR):
            
            if deep_sup:
                outputs = model(input)
                loss = sum(criterion(out, target) for out in outputs) / len(outputs)
                iou = iou_score(outputs[-1], target)
            else: # standard single-output case
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        avg_meters['loss'].update(loss.item() if isinstance(loss, torch.Tensor) else loss, input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        # Progress bar configs:::
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)    
        
    pbar.close()

    # Return a dictionary of average loss and IoU values
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
