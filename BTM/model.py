from transformers import RobertaTokenizer, RobertaModel
from transformers import DeiTFeatureExtractor, DeiTModel

import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import pdb

class BaseModel(nn.Module):
    def __init__(self,):
        super(BaseModel, self).__init__()        
        """RoBERTa setting"""
        text_model_path = "roberta-large" # '/data/project/rw/rung/02_source/model/roberta-large'
        self.text_model = RobertaModel.from_pretrained(text_model_path)
        self.text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
        special_token_list = ['[USER]', '[SYSTEM]']
        special_tokens = {'additional_special_tokens': special_token_list}
        self.text_tokenizer.add_special_tokens(special_tokens)
        self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        
        system_pos = self.text_tokenizer.additional_special_tokens.index('[SYSTEM]')
        self.system_token_id = self.text_tokenizer.additional_special_tokens_ids[system_pos]        
        
        """Deit setting"""
        image_model_path = "facebook/deit-base-distilled-patch16-224" # '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224'
        
        self.image_model = DeiTModel.from_pretrained(image_model_path, add_pooling_layer=False)
        
        """similarity"""
        self.hid_dim = 128
        self.text2hid = nn.Linear(self.text_model.config.hidden_size, self.hid_dim) # (1024, 128) self.text_model.config.hidden_size
        self.visual2hid = nn.Linear(self.image_model.config.hidden_size, self.hid_dim) # (768, 128) self.image_model.config.hidden_size          
        
        """score"""
        self.softmax = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sigmoid = nn.Sigmoid()
        
    def imgfea2rep(self, image_features):
        patch_represent = self.image_model(image_features)['last_hidden_state'] # (1, 196, 768)
        image_represent = patch_represent[:,0,:] # (1, 768)
        return image_represent

    def forward(self, text_tokens, bg_features):
        """ meta text projection"""
        text_logits = self.text_model(text_tokens).last_hidden_state # (batch, text_len, 1024)
        text_represent = text_logits[:,0,:] # (batch, 1024)
        t_h = self.text2hid(text_represent) # (batch, 128)
            
        """ object projection """
        bg_represent = self.imgfea2rep(bg_features) # (1, 768)
        bg_h = self.visual2hid(bg_represent) # (1, 128)
        
        """ meta-object score """
        t2bg_score = self.sigmoid(self.cos(t_h, bg_h) * 100) # (batch)
        return t2bg_score
            
        


        