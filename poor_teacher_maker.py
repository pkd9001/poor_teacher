# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:54:49 2023

@author: user
"""

# poor teacher maker

import torch
from transformers import BertModel, BertConfig, AutoTokenizer

def poor_teacher(num_layer, path):
    
    # BERT-base 모델 로드
    config = BertConfig.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased', config=config)
    
    # 모델의 레이어 수를 줄이기
    model.config.num_hidden_layers = num_layer
    model.encoder.layer = torch.nn.ModuleList([layer for i,  # layer에 따라 i < n 수치 정의
                                               layer in enumerate(model.encoder.layer) if i < num_layer])
    
    # 모델의 가중치 저장
    layer_num = str(num_layer)
    model_path = 'bert-base-' + layer_num
    model.save_pretrained('./'+ path + '/' + model_path)
    
    return print('save model name :' + model_path)