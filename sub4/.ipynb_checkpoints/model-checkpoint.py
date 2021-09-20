from transformers import GPT2Tokenizer, GPT2Model
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from transformers import DeiTFeatureExtractor, DeiTModel

import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import pdb

class BaseModel(nn.Module):
    def __init__(self, model_type, post):
        super(BaseModel, self).__init__()
        self.model_type = model_type
        if model_type == "gpt2-large":
            """GPT2 setting"""
            text_model_path = "gpt2-large" # '/data/project/rw/rung/02_source/model/gpt2-large'
            self.text_model = GPT2Model.from_pretrained(text_model_path)
            self.text_tokenizer = GPT2Tokenizer.from_pretrained(text_model_path)
            
        special_token_list = ['[USER]', '[SYSTEM]', '[RES]', '[META]']
        special_tokens = {'additional_special_tokens': special_token_list, 'pad_token': '[PAD]'}
        self.text_tokenizer.add_special_tokens(special_tokens)
        self.text_model.resize_token_embeddings(len(self.text_tokenizer))
        
        self.vocab_size = self.text_model.vocab_size
        
        system_pos = self.text_tokenizer.additional_special_tokens.index('[SYSTEM]')
        self.system_token_id = self.text_tokenizer.additional_special_tokens_ids[system_pos]        
        
        """Deit setting"""
        image_model_path = "facebook/deit-base-distilled-patch16-224" # '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224'
        self.image_model = DeiTModel.from_pretrained(image_model_path, add_pooling_layer=False)
        if post:
            post_path = "../ITM/post_training/all"
            post_model = os.path.join(post_path, 'model.pt')
            checkpoint = torch.load(post_model)
            self.image_model.load_state_dict(checkpoint, strict=False)
            print('Post-trained Model Loading!!')
        
        """ decoder """
        self.visual2hid = nn.Linear(self.image_model.config.hidden_size, self.text_model.config.hidden_size) # (768, 1280)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.text_model.config.hidden_size, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=4)
        self.hid2vocab = nn.Linear(self.text_model.config.hidden_size, self.vocab_size) # (1280, 50261)
        
        """multi-task"""
        
    def imgfea2rep(self, image_features):
        patch_represent = self.image_model(image_features)['last_hidden_state'] # (1, 196, 768)
        image_represent = patch_represent[:,0,:] # (1, 768)
        return image_represent
    
    def backfea2rep(self, bg_features):
        patch_represent = self.back_model(bg_features)['last_hidden_state'] # (1, 196, 768)
        image_represent = patch_represent[:,0,:] # (1, 768)
        return image_represent        

    def forward(self, batch_tokens_list, batch_obj_features_list, args):
        obj = args.object
        
        batch_decoder_outs = []
        for input_tokens, obj_features in zip(batch_tokens_list, batch_obj_features_list):
            """ dialogue text projection"""
            if self.model_type == 'gpt2-large':
                text_logits = self.text_model(input_tokens).last_hidden_state # (1, text_len, 1280)
            else: # bart-arge
                text_logits = self.text_model(input_tokens).encoder_last_hidden_state # (1, text_len, 1024) 

            """ object projection """
            if obj and (type(obj_features)!=type(False)):
                obj_logits = []
                obj_represent = self.imgfea2rep(obj_features) # (obj_num, 768)
                o_h = self.visual2hid(obj_represent) # (obj_num, 1280)
                obj_logits = o_h.unsqueeze(0) # (1, obj_num, 1280)
            else:
                obj_logits = torch.zeros(1,1,self.text_model.config.hidden_size).cuda()
                
            """ decoder """            
            obj_mean = torch.mean(obj_logits, 1).unsqueeze(1) # (1, 1, 1280)
            obj_mean = obj_mean.repeat(1, text_logits.shape[1], 1) # (1, text_len, 1280)
            decoder_out = text_logits + obj_mean # (1, text_len, 1280)
            
            batch_decoder_outs.append(self.hid2vocab(decoder_out).squeeze(0)) # (text_len, vocab_num)
            
        return batch_decoder_outs
        
        
            
        


        