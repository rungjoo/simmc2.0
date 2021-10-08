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
    def __init__(self):
        super(BaseModel, self).__init__()        
        """RoBERTa setting"""
        text_model_path = "roberta-large" # '/data/project/rw/rung/02_source/model/roberta-large' # 
        self.text_model = RobertaModel.from_pretrained(text_model_path)
        self.text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
        special_token_list = ['[USER]', '[SYSTEM]']
        special_tokens = {'additional_special_tokens': special_token_list}
        self.text_tokenizer.add_special_tokens(special_tokens)
        self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        
        system_pos = self.text_tokenizer.additional_special_tokens.index('[SYSTEM]')
        self.system_token_id = self.text_tokenizer.additional_special_tokens_ids[system_pos]        
        
        self.hid_dim = 128
        self.text2hid = nn.Linear(self.text_model.config.hidden_size, self.hid_dim) # (1024, 128) self.text_model.config.hidden_size        
                
        """multi-task"""
        # system matching
        self.W_system = nn.Linear(self.text_model.config.hidden_size, 2) # (1024, 2)        

    def forward(self, text_tokens):
        """ dialogue text projection"""
        text_logits = self.text_model(text_tokens).last_hidden_state # (batch, text_len, 1024)
        text_represent = text_logits[:,0,:] # (batch, 1024)        
        t_h = self.text2hid(text_represent) # (batch, 128)
        
        """ system matching """
        system_logits_list = []
        for batch, text_token in enumerate(text_tokens):
            # text_token: (text_len)
            system_pos_list = []
            for system_pos, token in enumerate(text_token.tolist()):
                if token == self.system_token_id:
                    system_pos_list.append(system_pos)        

            system_hidden = []
            for system_pos in system_pos_list:
                system_hidden.append(text_logits[batch:batch+1,system_pos,:]) # [(1, 1024), ..., ]

            if len(system_hidden) == 0:
                system_logits_list.append([])
            else:
                system_hiddens = torch.cat(system_hidden, 0) # (len, 1024)
                system_logits = self.W_system(system_hiddens) # (len, 2)
                system_logits_list.append(system_logits) # [(len1,2), (len2,2)]
            
        return system_logits_list
            
        


        