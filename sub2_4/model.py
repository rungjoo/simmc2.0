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
    def __init__(self, post_back):
        super(BaseModel, self).__init__()        
        """RoBERTa setting"""
        text_model_path = '/data/project/rw/rung/02_source/model/roberta-large' # "roberta-large" # 
        self.text_model = RobertaModel.from_pretrained(text_model_path)
        self.text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
        special_token_list = ['[USER]', '[SYSTEM]']
        special_tokens = {'additional_special_tokens': special_token_list}
        self.text_tokenizer.add_special_tokens(special_tokens)
        self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        
        system_pos = self.text_tokenizer.additional_special_tokens.index('[SYSTEM]')
        self.system_token_id = self.text_tokenizer.additional_special_tokens_ids[system_pos]        
        
        """Deit setting"""
        image_model_path = '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224' # "facebook/deit-base-distilled-patch16-224" # 
        self.image_model = DeiTModel.from_pretrained(image_model_path, add_pooling_layer=False)
        
        if post_back:            
            self.back_model = DeiTModel.from_pretrained(image_model_path, add_pooling_layer=False)
            post_back_path = "../BTM/bg_model"
            post_back_model = os.path.join(post_back_path, 'model.pt')
            checkpoint = torch.load(post_back_model)
            self.back_model.load_state_dict(checkpoint, strict=False)
            print('PostBack-trained Model Loading!!')
        
        """similarity"""
        self.hid_dim = 128
        self.text2hid = nn.Linear(self.text_model.config.hidden_size, self.hid_dim) # (1024, 128) self.text_model.config.hidden_size
        self.visual2hid = nn.Linear(self.image_model.config.hidden_size, self.hid_dim) # (768, 128) self.image_model.config.hidden_size
        self.bg2hid = nn.Linear(self.image_model.config.hidden_size, self.hid_dim) # (768, 128) self.image_model.config.hidden_size
                
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
    
    def backfea2rep(self, bg_features):
        patch_represent = self.back_model(bg_features)['last_hidden_state'] # (1, 196, 768)
        image_represent = patch_represent[:,0,:] # (1, 768)
        return image_represent        

    def forward(self, text_tokens, meta_tokens, object_features, background_features, score_type, meta, system_matching, utt_category, background, post_back):
        """ dialogue text projection"""
        text_logits = self.text_model(text_tokens).last_hidden_state # (batch, text_len, 1024)
        text_represent = text_logits[:,0,:] # (batch, 1024)        
        t_h = self.text2hid(text_represent) # (batch, 128)
        
        """ meta text projection"""
        if meta:
            meta_logits = self.text_model(meta_tokens).last_hidden_state # (batch, text_len, 1024)
            meta_represent = meta_logits[:,0,:] # (batch, 1024)
            m_h = self.text2hid(meta_represent) # (batch, 128)
            
        """ object projection """
        image_represent = self.imgfea2rep(object_features) # (batch, 768)
        o_h = self.visual2hid(image_represent) # (batch, 128)
        
        """ background proejction """
        if background:
            if post_back:
                bg_represent = self.backfea2rep(background_features) # (batch, 768)
            else:
                bg_represent = self.imgfea2rep(background_features) # (batch, 768)
            bg_h = self.bg2hid(bg_represent) # (batch, 128)
        else:
            bg_h = torch.zeros(o_h.shape).cuda()
        
        """ total visual represent vector """
        v_h = o_h + bg_h
        
        """ system matching """
        system_logits_list = []
        if system_matching:
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
        else:
            system_logits_list.append([])
            
        """ utterance category prediction"""
        if utt_category:
            utt_category_logits = self.W_category(t_h) # (batch, 4)
        else:
            utt_category_logits = torch.zeros([t_h.shape[0],4])

        """ dialog-object score """
        if score_type == 'cos':
            t2v_score = (self.cos(t_h, v_h)+1)/2
        elif score_type == 'sigmoid':
            t2v_score = self.sigmoid(self.cos(t_h, v_h) * 100) # (batch)
        else: ## concat
            final_h = torch.cat([t_h, v_h], 1) # (1, 128*x)
            t2v_logit = self.concat2score2(final_h) # (1,2)
            t2v_score = self.softmax(t2v_logit)[:,1]
        
        """ meta-object score """            
        if meta:
            m2v_score = self.sigmoid(self.cos(m_h, v_h) * 100) # (batch)
            return t2v_score, m2v_score, system_logits_list, utt_category_logits
        else:
            m2v_score = torch.zeros(t2v_score.shape)
            return t2v_score, m2v_score, system_logits_list, utt_category_logits
            
        


        