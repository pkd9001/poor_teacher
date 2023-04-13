# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:53:20 2022

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from customdataset import CustomDataset
import numpy as np
import random
import os
from datasets import load_dataset

# from torch.utils.data.distributed import DistributedSampler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dataloader(batch_size, num_workers, max_length):
    
    dataset = load_dataset('glue','mrpc')
    
    train = dataset['train']
    train = pd.DataFrame(train)
    train_data = train[['sentence1', 'sentence2','label']]
    
    val = dataset['validation']
    val = pd.DataFrame(val)
    val_data = val[['sentence1', 'sentence2','label']]
    
    train_dataset = CustomDataset(train_data, max_length)
    val_dataset = CustomDataset(val_data, max_length)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers)
    
    return train_loader, val_loader
    
class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def seed_everything(seed:int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss(reduction="batchmean")(F.log_softmax(y/T, dim=-1),
                          F.softmax(teacher_scores/T, dim=-1)) * (T*T) * alpha + F.cross_entropy(y,labels) * (1.-alpha)

def distillation_TL(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss(reduction="batchmean")(F.log_softmax(y/T, dim=-1),
                          F.softmax(teacher_scores/T, dim=-1)) * (T*T) * alpha + F.cross_entropy(y,labels) * alpha

def final_loss(outputs, labels, teacher_scores, teacher_hidden, poor_teacher_hidden, student_hidden, T, alpha):

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    TL = triplet_loss(student_hidden, teacher_hidden, poor_teacher_hidden)
    
    loss = distillation_TL(outputs, labels, teacher_scores, T, alpha) + TL * alpha
        
    return loss