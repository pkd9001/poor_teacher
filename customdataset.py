# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:54:31 2022

@author: user
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer

class CustomDataset:
    def __init__(self, dataset=None, max_length = 512):
        self.dataset = dataset
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
  
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        dataset = self.dataset.iloc[idx]
        datas = self.tokenizer(dataset['sentence1'], dataset['sentence2'],
                               padding='max_length',
                               max_length=self.max_length,
                               truncation=True,
                               )
        input_ids = torch.tensor(datas['input_ids'])
        token_type_ids = torch.tensor(datas['token_type_ids'])
        attention_mask = torch.tensor(datas['attention_mask'])
        labels = torch.tensor(self.dataset.iloc[idx]['label'])
        
        return input_ids, token_type_ids, attention_mask, labels