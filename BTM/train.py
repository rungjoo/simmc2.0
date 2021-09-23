# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup

import pdb
import argparse, logging
import glob, random

from model import BaseModel
from dataset import post_loader
from utils import img2feature

from transformers import RobertaTokenizer
text_model_path = "roberta-large" # '/data/project/rw/rung/02_source/model/roberta-large'
model_text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
special_token_list = ['[USER]', '[SYSTEM]']
special_tokens = {'additional_special_tokens': special_token_list}
model_text_tokenizer.add_special_tokens(special_tokens)

from transformers import DeiTFeatureExtractor
image_model_path = "facebook/deit-base-distilled-patch16-224" # '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224'
image_feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)

def clsLoss(batch_t2v_score, labels):
    """
    batch_t2v_score: [batch]
    labels: [batch]
    """
    loss = nn.BCELoss()
    
    loss_val = loss(batch_t2v_score, labels)
    
    return loss_val

def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    
def CalPER(model, dataloader):
    threshold = 0.5
    correct = 0
    total_cnt = 0
    pred_list, label_list = [], []
    for i_batch, (batch_text_tokens, batch_bg_features, batch_match_labels) in enumerate(tqdm(dataloader, desc='eval_iteration')):
        batch_text_tokens = batch_text_tokens.cuda() # (batch, len)
        batch_bg_features = batch_bg_features.type('torch.FloatTensor').cuda() # (1, 3, 224, 224)            
        batch_match_labels = batch_match_labels # (batch)
        
        batch_t2bg_score = model(batch_text_tokens, batch_bg_features)
        
        t2bg_score = batch_t2bg_score.tolist()
        match_labels = batch_match_labels.tolist()
        
        for score, label in zip(t2bg_score, match_labels):
            if score > threshold:
                pred = 1
            else:
                pred = 0
                
            total_cnt += 1
            if pred == label:
                correct += 1
            pred_list.append(pred)
            label_list.append(label)
    precision, recall, f1, _ = precision_recall_fscore_support(label_list, pred_list, labels=[1], average='weighted')
    
    return f1*100, correct/total_cnt*100
    

def make_batch(sessions):
    # dict_keys(['text', 'background', 'neg_background'])
    backgrounds = [session['background'] for session in sessions]
    neg_backgrounds = [session['neg_background'] for session in sessions]
    texts = [session['text'] for session in sessions]

    assert len(backgrounds) == len(neg_backgrounds)
    assert len(backgrounds) == len(texts)
    
    backgrounds_features = []
    match_labels = []
    total_text_strs = []
    for background, neg_background, text in zip(backgrounds, neg_backgrounds, texts):        
        backgrounds_features.append(img2feature(background, image_feature_extractor))
        backgrounds_features.append(img2feature(neg_background, image_feature_extractor))
        total_text_strs.append(text)
        
        # 1:1
        match_labels.append(1)
        match_labels.append(0)
            
    batch_text_tokens = model_text_tokenizer(total_text_strs, padding='longest', return_tensors='pt').input_ids
    batch_bg_features = torch.cat(backgrounds_features,0)    
    batch_match_labels = torch.tensor(match_labels)
    
    return batch_text_tokens, batch_bg_features, batch_match_labels

def main():
    gpu_name = args.device
    device = torch.device('cuda:'+str(gpu_name))
    
    """save & log path"""
    model_type = args.model_type
    score_type = args.score
    save_path = 'bg_model'
    print("###Save Path### ", save_path)
    
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""
    model = BaseModel().cuda()
    model.train()    
    
    """dataset Loading"""
    image_obj_path = "../res/image_obj.pickle"
    description_path = "../data/public/*scene*"
    fashion_path = '../data/fashion_prefab_metadata_all.json'
    furniture_path = '../data/furniture_prefab_metadata_all.json'
    
    train_path = '../data/simmc2_dials_dstc10_train.json'    
    dev_path = '../data/simmc2_dials_dstc10_dev.json'
    devtest_path = '../data/simmc2_dials_dstc10_devtest.json'
            
    batch_size = args.batch
    print('batch size: ', batch_size)
    
    train_dataset = post_loader(train_path, image_obj_path, description_path, fashion_path, furniture_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=make_batch)
    
    dev_dataset = post_loader(dev_path, image_obj_path, description_path, fashion_path, furniture_path)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=make_batch)
    
    devtest_dataset = post_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path)
    devtest_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    """Training Parameter Setting"""    
    training_epochs = args.epoch
    print('Training Epochs: ', str(training_epochs))
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)    
    
    """Training"""    
    logger.info('########################################')
    # model.text_tokenizer.save_pretrained(save_path)
    best_dev_f1, best_epoch = -1, 0
    print("Data Num ## ", len(train_loader))
    for epoch in range(training_epochs):
        model.train()
        for i_batch, (batch_text_tokens, batch_bg_features, batch_match_labels) in enumerate(tqdm(train_loader, desc='train_iteration')):
            batch_text_tokens = batch_text_tokens.cuda() # (batch, len)
            batch_bg_features = batch_bg_features.type('torch.FloatTensor').cuda() # (1, 3, 224, 224)            
            batch_match_labels = batch_match_labels.type('torch.FloatTensor').cuda() # (batch)
            
            """Model Training"""
            batch_t2bg_score = model(batch_text_tokens, batch_bg_features)

            loss_val = clsLoss(batch_t2bg_score, batch_match_labels)

            optimizer.zero_grad()
            loss_val.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            
        for i_batch, (batch_text_tokens, batch_bg_features, batch_match_labels) in enumerate(tqdm(dev_loader, desc='dev_iteration')):
            batch_text_tokens = batch_text_tokens.cuda() # (batch, len)
            batch_bg_features = batch_bg_features.type('torch.FloatTensor').cuda() # (1, 3, 224, 224)            
            batch_match_labels = batch_match_labels.type('torch.FloatTensor').cuda() # (batch)
            
            """Model Training"""
            batch_t2bg_score = model(batch_text_tokens, batch_bg_features)

            loss_val = clsLoss(batch_t2bg_score, batch_match_labels)

            optimizer.zero_grad()
            loss_val.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            
        """ Score and Save"""        
        model.eval()
        devf1, devacc = CalPER(model, devtest_loader)
        logger.info("Epoch: {}, DevTestf1: {}, Acc: {}".format(epoch, devf1, devacc))
        if devf1 > best_dev_f1:
            _SaveModel(model, save_path)
            best_dev_f1 = devf1            
            best_epoch = epoch
    logger.info("Best Epoch: {}, DevTestf1: {}".format(best_epoch, best_dev_f1))
            

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Taks2" )
    parser.add_argument( "--device", type=int, help = "cuda", default = 0) 
    
    parser.add_argument( "--epoch", type=int, help = 'training epochs', default = 5) # 12 for iemocap
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)    
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    parser.add_argument( "--model_type", help = "large", default = 'roberta-large') # large
    parser.add_argument( "--batch", type=int, help = "training batch size", default =1) 
    parser.add_argument( "--score", type=str, help = "cosine norm or sigmoid or concat", default = 'sigmoid') # cos or sigmoid
    
    parser.add_argument('--relation', action='store_true', help='use around object features')
    parser.add_argument('--background', action='store_true', help='use background image features')
    
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()