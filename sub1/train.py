# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.nn.functional import softmax

import pdb
import argparse, logging
import glob

from model import BaseModel
from dataset import task1_loader

from utils import img2feature

from transformers import RobertaTokenizer
text_model_path = '/data/project/rw/rung/02_source/model/roberta-large' # "roberta-large" # 
model_text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
special_token_list = ['[USER]', '[SYSTEM]']
special_tokens = {'additional_special_tokens': special_token_list}
model_text_tokenizer.add_special_tokens(special_tokens)

from transformers import DeiTFeatureExtractor
image_model_path = '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224' # "facebook/deit-base-distilled-patch16-224" # 
image_feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)

def CELoss(pred_logits, labels):
    """
    pred_logits: [batch, clsNum]
    labels: [clsNum]
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_logits, labels)
    return loss_val


def _CalACC(model, dataloader, args):
    model.eval()
    
    domain, background= args.domain, args.background
    obj = args.object
    
    disamb_correct, domain_correct = 0, 0    
    with torch.no_grad():
        for i_batch, (input_sample) in enumerate(tqdm(dataloader, desc='evaluation')):
            batch_tokens, disamb_label, domain_label, batch_background_features, batch_object_features = input_sample
            batch_tokens = batch_tokens.cuda() # (1, len)
            batch_background_features = batch_background_features.type('torch.FloatTensor').cuda() # [1, 3, 224, 224]
            batch_object_features = [x.type('torch.FloatTensor').cuda() for x in batch_object_features] # [[objnum, 3, 224, 224], ..., ]
            
            """ prediction """                
            disamb_logit, domain_logit = model(batch_tokens, batch_background_features, batch_object_features, domain, background, obj)
            
            """Calculation"""
            pred_disamb = disamb_logit.argmax(1).item()
            if pred_disamb == disamb_label.item():
                disamb_correct += 1
            
            if domain:
                pred_domain = domain_logit.argmax(1).item()
                if pred_domain == domain_label.item():
                    domain_correct += 1
    
    acc = disamb_correct/len(dataloader)*100
    if domain:
        domain_acc = domain_correct/len(dataloader)*100
    else:
        domain_acc = 0
    
    return acc, domain_acc

def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))

def make_batch(sessions):
    # dict_keys(['input', 'label', 'domain_label', 'pre_system_utts', 'object', 'background'])
    input_strs = [session['input'] for session in sessions]
    disamb_labels = [session['label'] for session in sessions]
    domain_labels = [session['domain_label'] for session in sessions]
    
    pre_system_utts_list = [session['pre_system_utts'] for session in sessions]
    system_object_ids_list = [session['system_object'] for session in sessions]    
    object_datas_list = [session['object'] for session in sessions]
    backgrounds = [session['background'] for session in sessions]
    
    batch_tokens = model_text_tokenizer(input_strs, padding='longest', add_special_tokens=False).input_ids # (batch, text_len, 1024)
    batch_token_list = []
    for batch_token in batch_tokens:
        batch_token = [model_text_tokenizer.cls_token_id] + batch_token[-model_text_tokenizer.model_max_length+1:]
        batch_token_list.append(torch.tensor(batch_token).unsqueeze(0))
    batch_tokens = torch.cat(batch_token_list, 0)
    
    batch_disamb_labels = torch.tensor(disamb_labels)
    batch_domain_labels = torch.tensor(domain_labels)
    
    background_features = []
    for background_list in backgrounds:
        bg_features = 0
        for background in background_list:
            bg_features += img2feature(background, image_feature_extractor)
        background_features.append(bg_features)
    batch_background_features = torch.cat(background_features, 0)
    
    assert len(system_object_ids_list) == len(object_datas_list)
    total_object_features = []
    for object_datas, system_object_ids in zip(object_datas_list, system_object_ids_list):
        object_features = []
        for object_id, object_data in object_datas.items():
            if object_id in system_object_ids:
                object_visual = object_data['visual']
                object_feature = img2feature(object_visual, image_feature_extractor)
                object_features.append(object_feature)
        if len(object_features) > 0:
            total_object_features.append(torch.cat(object_features, 0))
        else:
            total_object_features.append(torch.tensor([0]))
    
    return batch_tokens, batch_disamb_labels, batch_domain_labels, batch_background_features, total_object_features
    
def main():    
    """save & log path"""
    model_type = args.model_type
    current = args.current
    domain = args.domain
    if domain:
        domain_type = 'domain_use'
    else:
        domain_type = 'domain_no_use'
    background = args.background
    if background:
        background_type = 'background_use'
    else:
        background_type = 'background_no_use'
    obj = args.object
    if background:
        obj_type = 'object_use'
    else:
        obj_type = 'object_no_use'
    post = args.post
    if post:
        post_type = 'post_use'
    else:
        post_type = 'post_no_use'      
    # save_path = os.path.join(model_type+'_models', current, domain_type, background_type, obj_type, post_type)
    save_path = 'results'
    print("###Save Path### ", save_path)
    print("use history utterance?: ", current)
    print("domain prediction learning?: ", domain)
    print("background image use?: ", background)
    print("object image use?: ", obj)
    
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""
    model = BaseModel(post).cuda()
    model.train()
    
    """dataset Loading"""
    image_obj_path = "../res/image_obj.pickle"
    description_path = "../data/public/*"
    fashion_path = '../data/fashion_prefab_metadata_all.json'
    furniture_path = '../data/furniture_prefab_metadata_all.json'

    train_path = '../data/simmc2_dials_dstc10_train.json'
    dev_path = '../data/simmc2_dials_dstc10_dev.json'
    devtest_path = '../data/simmc2_dials_dstc10_devtest.json'
            
    batch_size = args.batch
    print('batch size: ', batch_size)
    
    train_dataset = task1_loader(train_path, image_obj_path, description_path, fashion_path, furniture_path, current)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    dev_dataset = task1_loader(dev_path, image_obj_path, description_path, fashion_path, furniture_path, current)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    devtest_dataset = task1_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path, current)
    devtest_loader = DataLoader(devtest_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
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
    best_dev_acc, best_epoch = 0, 0
    total_disamb_loss, total_domain_loss = 0, 0
    for epoch in range(training_epochs):
        model.train()
        for i_batch, (input_sample) in enumerate(tqdm(train_loader, desc='iteration')):
            batch_tokens, disamb_labels, domain_labels, batch_background_features, total_object_features = input_sample
            batch_tokens = batch_tokens.cuda() # (1, len)
            disamb_labels = disamb_labels.cuda() # [1]
            domain_labels = domain_labels.cuda() # [1]
            batch_background_features = batch_background_features.type('torch.FloatTensor') #.cuda() # [1, 3, 224, 224]
            batch_object_features = [x.type('torch.FloatTensor').cuda() for x in total_object_features] # [[objnum, 3, 224, 224], ..., ]
            total_obj_num = 0
            for batch_object_feature in batch_object_features:
                total_obj_num += batch_object_feature.shape[0]
            if total_obj_num > 100:
                print(total_obj_num)
            
            """ model training """
            disamb_logits, domain_logits = model(batch_tokens, batch_background_features, batch_object_features, domain, background, obj)
                
            disamb_loss_val = CELoss(disamb_logits, disamb_labels)
            total_disamb_loss += disamb_loss_val.item()
            
            if domain:
                domain_loss_val = CELoss(domain_logits, domain_labels)
                total_domain_loss += domain_loss_val.item()
            else:
                domain_loss_val = 0
                total_domain_loss += domain_loss_val
            if (i_batch+1) % 4000 == 0:
                print("i_batch: {}, disamb_loss: {}, domain_loss: {}".format(i_batch+1, total_disamb_loss/(i_batch+1), total_domain_loss/(i_batch+1)))
                
            loss_val = disamb_loss_val + domain_loss_val
            
            optimizer.zero_grad()
            loss_val.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            
        """ Score and Save"""
        model.eval()
        devAcc, devDomainAcc = _CalACC(model, dev_loader, args)
        logger.info("Epoch: {}, DevAcc: {}, DevDomainAcc: {}".format(epoch, devAcc, devDomainAcc))
        if devAcc > best_dev_acc:
            _SaveModel(model, 'model')
            best_dev_acc = devAcc
            
            best_epoch = epoch
            devtestAcc, devtestDomainACC = _CalACC(model, devtest_loader, args)
            logger.info("DevTestAcc: {}, DevTestDomainAcc: {}".format(devtestAcc, devtestDomainACC))
    
    logger.info("")
    logger.info("Epoch: {}, DevTestAcc: {}, DevTestDomainAcc: {}".format(best_epoch, devtestAcc, devtestDomainACC))
            

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Task1" )
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 10)
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-5) # 1e-5
    parser.add_argument( "--model_type", help = "base", default = 'roberta-large') # base
    parser.add_argument( "--batch", type=int, help = "training batch size", default =1) # base
    parser.add_argument( "--current", type=str, help = 'only use current utt / system current / context', default = 'context') # current or sys_current
        
    parser.add_argument('--domain', action='store_true', help='domain multi-task learning')
    parser.add_argument('--background', action='store_true', help='use background image features')
    parser.add_argument('--object', action='store_true', help='use object features')
    parser.add_argument('--post', action='store_true', help='post-trained model')
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()