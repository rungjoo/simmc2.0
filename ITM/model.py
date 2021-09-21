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
        # self.feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)
        
        """similarity"""
        self.hid_dim = 128
        self.text2hid = nn.Linear(self.text_model.config.hidden_size, self.hid_dim) # (1024, 128) self.text_model.config.hidden_size
        self.visual2hid = nn.Linear(self.image_model.config.hidden_size, self.hid_dim) # (768, 128) self.image_model.config.hidden_size
                
        """multi-task"""
        # utterance category prediction
        self.W_category = nn.Linear(self.hid_dim, 4) # (128, 4)
        # system matching
        self.W_system = nn.Linear(self.text_model.config.hidden_size, 2) # (1024, 2)        
        
        """score"""
        self.softmax = nn.Softmax(dim=1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sigmoid = nn.Sigmoid()
        
    def imgfea2rep(self, image_features):
        patch_represent = self.image_model(image_features)['last_hidden_state'] # (1, 196, 768)
        image_represent = patch_represent[:,0,:] # (1, 768)
        return image_represent

    def forward(self, meta_tokens, object_features):
        """ meta text projection"""
        meta_logits = self.text_model(meta_tokens).last_hidden_state # (batch, text_len, 1024)
        meta_represent = meta_logits[:,0,:] # (batch, 1024)
        m_h = self.text2hid(meta_represent) # (batch, 128)
            
        """ object projection """
        image_represent = self.imgfea2rep(object_features) # (1, 768)
        v_h = self.visual2hid(image_represent) # (1, 128)
        
        """ meta-object score """
        m2v_score = self.sigmoid(self.cos(m_h, v_h) * 100) # (batch)
        return m2v_score
            
        


        