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
    def __init__(self, post):
        super(BaseModel, self).__init__()        
        """RoBERTa setting"""
        model_path = "roberta-large" # '/data/project/rw/rung/02_source/model/roberta-large' # 
        self.text_model = RobertaModel.from_pretrained(model_path)
        self.text_tokenizer = RobertaTokenizer.from_pretrained(model_path)
        special_token_list = ['[USER]', '[SYSTEM]']
        special_tokens = {'additional_special_tokens': special_token_list}
        self.text_tokenizer.add_special_tokens(special_tokens)
        self.text_model.resize_token_embeddings(len(self.text_tokenizer))        
        
        """Deit setting"""
        image_model_path = "facebook/deit-base-distilled-patch16-224" # '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224' # 
        self.image_model = DeiTModel.from_pretrained(image_model_path, add_pooling_layer=False)
        if post:
            post_path = "../ITM/post_training/all"
            post_model_path = os.path.join(post_path, 'model.pt')
            checkpoint = torch.load(post_model_path)
            self.image_model.load_state_dict(checkpoint, strict=False)
            print('Post-trained Model Loading!!')

        
        """projection"""
        self.hid_dim = 256
        
        self.text2hid = nn.Linear(self.text_model.config.hidden_size, self.hid_dim) # (1024, 768) self.text_model.config.hidden_size
        self.visual2hid = nn.Linear(self.image_model.config.hidden_size, self.hid_dim) # (768, 768) self.image_model.config.hidden_size
                
        self.text2Q = nn.Linear(self.hid_dim, self.hid_dim) # (768, 768) 
        self.visual2K = nn.Linear(self.hid_dim, self.hid_dim) # (768, 768)
        self.visual2V = nn.Linear(self.hid_dim, self.hid_dim) # (768, 768)
        
        self.multihead_attn = nn.MultiheadAttention(self.hid_dim, 4)        
        
        """ layer """
        self.W = nn.Linear(self.hid_dim, 2)
        self.W_domain = nn.Linear(self.hid_dim, 2)

    
    def text2visual_attn(self, t_h, v_h):
        """
        t_h: (batch, len, hid_dim)
        v_h: (objnum, 1, hid_dim)
        """        
        t_Q = self.text2Q(t_h).transpose(0,1) # (len, 1, hid_dim) (seq_len, batch, embedding)
        v_K = self.visual2K(v_h) # (objnum, 1 hid_dim)
        v_V = self.visual2V(v_h) # (objnum, 1, hid_dim)
        
        attn_output, attn_output_weights = self.multihead_attn(t_Q, v_K, v_V) # (len, 1, hid_dim)
        # attn_output: (len, 1, hid_dim)
        return attn_output[:,0,:] # (len, hid_dim)
        
    
    def imgfea2rep(self, image_features, first=False):
        patch_represent = self.image_model(image_features)['last_hidden_state'] # (1, 198, 768)
        image_represent = patch_represent # (1, 198, 768)
        if first:
            return image_represent[:,0,:] # (1, 768)
        else:
            return image_represent # (1, 198, 768)
        

    def forward(self, batch_tokens, batch_background_features, batch_object_features, domain, background, obj):
        """ text projection """
        text_logits = self.text_model(batch_tokens).last_hidden_state # (batch, text_len, 1024)
        # t_h = self.text2hid(text_logits)[:,0:1,:] # (batch, 1, hid_dim)
        t_h = self.text2hid(text_logits) # (batch, len, hid_dim)
                
#         if background:
#             """ background projection """
#             bg_represent = self.imgfea2rep(batch_background_features, first=True) # (batch, 768)
#             bg_h = self.visual2hid(bg_represent).unsqueeze(1) # (batch, 1, 256)

#             attn_out = self.text2visual_attn(t_h, bg_h) # (batch, hid_dim)
#         else:
#             attn_out = t_h.squeeze(1) # (batch, hid_dim)
        
        if obj:
            """ object projecion """
            batch_obj_out = []
            for batch, object_features in enumerate(batch_object_features):
                if object_features.shape[-1] == 1:
                    obj_mean = torch.zeros([1, self.hid_dim]).cuda()
                else:
                    obj_represent = self.imgfea2rep(object_features, first=True) # (objnum, 768)
                    obj_mean = torch.mean(obj_represent, 0).unsqueeze(0) # (1, 768)
                    obj_mean = self.visual2hid(obj_mean)
                batch_obj_out.append(obj_mean)
            obj_out = torch.cat(batch_obj_out, 0) # (batch, hid_dim)
        else:
            obj_out = t_h.squeeze(1) # (batch, hid_dim)
        
        # final_h = torch.cat([t_h[:,0,:], obj_out], 1) # (batch, hid_dim+768)
        final_h = t_h[:,0,:]+obj_out # (batch, hid_dim)
        """ prediction """
        disamb_logit = self.W(final_h) # (1, 2)        
        
        if domain:
            domain_logit = self.W_domain(final_h) # (1, 2)
        else:
            domain_logit = torch.zeros(1,2)
        return disamb_logit, domain_logit        