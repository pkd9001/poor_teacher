# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:54:54 2022

@author: user
"""

import torch
import torch.nn as nn

from transformers import (AdamW,
                          BertForSequenceClassification,
                          )
from transformers.optimization import get_linear_schedule_with_warmup

import numpy as np
import pandas as pd

from tqdm import tqdm
import re

import os
from custom_utils import *
from torchmetrics import F1Score
import gc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pretrain_student = "./student_model/bert-base-6"
pretrain_teacher = "distilbert-base-cased" # "bert-base-cased" 
pretrain_poor_teacher = "./poor_teacher/bert-base-1"

EPOCHS = 10
batch_size = 32
warmup_ratio = 0.1
max_grad_norm = 1
max_length = 512
num_workers = 0
smoothing = 0.0
T = 4.0
alpha = 0.5
alpha_TL = 1/3

num_start = 1
num_end = 6

train_loader, val_loader = dataloader(batch_size, num_workers, max_length)

total_steps = len(train_loader) * EPOCHS
warmup_step = int(total_steps * warmup_ratio)

loss_function = distillation, final_loss

for y in loss_function:
    for s in range(num_start, num_end):
        
        print(y)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        seed_everything(s)
    
        model = BertForSequenceClassification.from_pretrained(pretrain_student, num_labels=2,
                                                               output_hidden_states = True,
                                                              ).to(device)
        
        teacher_model = BertForSequenceClassification.from_pretrained(pretrain_teacher, num_labels=2,
                                                               output_hidden_states = True,
                                                              ).to(device)
        
        model_state_dict_teacher = torch.load("model/Epoch_5_loss_0.0014.pt")
        teacher_model.load_state_dict(model_state_dict_teacher)
        
        if y == distillation :
            pass
        else :
            poor_teacher_model = BertForSequenceClassification.from_pretrained(pretrain_poor_teacher, num_labels=2,
                                                                               output_hidden_states = True,
                                                                              ).to(device)
            model_state_dict_poor_teacher = torch.load("./poor_teacher/Epoch_1_loss_0.0028.pt")
            poor_teacher_model.load_state_dict(model_state_dict_poor_teacher)
        
        optimizer = AdamW(model.parameters(), lr=6e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_step,
                                                    num_training_steps=total_steps)

        f1 = F1Score(num_classes=2, task='binary').to(device)
        
        PATH = './model/'
        
        Train_Loss = []
        train_acc = []
        Test_acc = []
        f1_acc = []
        
        for i in range(EPOCHS):
            total_loss = 0.0
            correct_train = 0
            correct_eval = 0
            total_train = 0
            total_eval = 0
            
            f1_score = 0
        
            model.train()
            teacher_model.eval()
            if y == distillation :
                pass
            else :
                poor_teacher_model.eval()
            
            for batch in tqdm(train_loader):
                batch = tuple(v.to(device) for v in batch)
                input_ids, token_type_ids, attention_masks, labels = batch
                
                with torch.no_grad():
                    teacher_out = teacher_model(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=attention_masks,
                                                labels=labels)
                    teacher_logit = teacher_out[1]
                    teacher_hidden = teacher_out.hidden_states[-1]
                    
                    if y == distillation :
                        pass
                    else :
                        poor_teacher_out = poor_teacher_model(input_ids=input_ids,
                                                    token_type_ids=token_type_ids,
                                                    attention_mask=attention_masks,
                                                    labels=labels)
                        poor_teacher_logit = poor_teacher_out[1]
                        poor_teacher_hidden = poor_teacher_out.hidden_states[-1]
                
                out = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_masks,
                            labels=labels)
        
                logits = out[1]
                student_hidden_state = out.hidden_states[-1]
                
                if y == distillation:
                    loss_function = distillation(logits, labels, teacher_logit, T, alpha)
                else :
                    loss_function = final_loss(logits,
                                               labels,
                                               teacher_logit,
                                               teacher_hidden,
                                               poor_teacher_hidden,
                                               student_hidden_state,
                                               T,
                                               alpha_TL)
                
                loss = loss_function
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                  
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                correct_train += (predicted == labels).sum()
                total_train += len(labels)
        
            print("")
            print("epoch {} Train Loss {:.6f} train acc {:.6f}".format(i+1,
                                                               torch.true_divide(total_loss, total_train),
                                                               torch.true_divide(correct_train, total_train)))
            print("")
        
            # torch.save(model.state_dict(), PATH + 'Epoch_{}_loss_{:.4f}.pt'.format(i+1,
            #                                                                        torch.true_divide(total_loss,
            #                                                                                          total_train)))
            
            model.eval()
            for batch in tqdm(val_loader):
                batch = tuple(v.to(device) for v in batch)
                input_ids, token_type_ids, attention_masks, labels = batch
                with torch.no_grad():
                    out = model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_masks,
                                labels=labels
                                )
                logits = out[1]
                _, predicted = torch.max(logits, 1)
                correct_eval += (predicted == labels).sum()
                total_eval += len(labels)
                
                f1_score += f1(predicted, labels)
            f1_score /= len(val_loader)
            
            print("")
            print("epoch {} Test acc {:.6f} F1 score {:.6f}".format(i+1,
                                                                    torch.true_divide(correct_eval, total_eval),
                                                                    f1_score))
            print("")
            
            Train_Loss.append(torch.true_divide(total_loss, total_train).item())
            train_acc.append(torch.true_divide(correct_train, total_train).item())
            Test_acc.append(torch.true_divide(correct_eval, total_eval).item())
            f1_acc.append(f1_score.item())
            
        Train_Loss = pd.DataFrame(Train_Loss)
        train_acc = pd.DataFrame(train_acc)
        Test_acc = pd.DataFrame(Test_acc)
        f1_acc = pd.DataFrame(f1_acc)
        
        Train_Loss.columns = ['Train_Loss']
        train_acc.columns = ['train_acc']
        Test_acc.columns = ['Test_acc']
        f1_acc.columns = ['f1_acc']
        
        x_list = pd.concat([Train_Loss, train_acc, Test_acc, f1_acc],axis=1)
        
        print(x_list)
        
        num_seed = str(s)
        
        print("")
        print("seed number :" + num_seed)
        print("used loss : " + str(y == distillation) + ", distillation loss if 'True'")
        print("")
        
        if y == distillation :
            x_list.to_excel('./normal KD seed/{} seed {}.xlsx'.format("normal KD", num_seed))
        else :
            x_list.to_excel('./KD-FT-TL seed/{} seed {}.xlsx'.format("KD-FT-TL", num_seed))
            