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
import glob, json

from model import BaseModel
from dataset import task1_loader
from utils import img2feature

from transformers import RobertaTokenizer
text_model_path = "roberta-large" # '/data/project/rw/rung/02_source/model/roberta-large' # 
model_text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
special_token_list = ['[USER]', '[SYSTEM]']
special_tokens = {'additional_special_tokens': special_token_list}
model_text_tokenizer.add_special_tokens(special_tokens)

from transformers import DeiTFeatureExtractor
image_model_path = "facebook/deit-base-distilled-patch16-224" # '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224' # 
image_feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)

def make_batch(sessions):
    input_strs = [session['input'] for session in sessions]
    disamb_labels = [session['label'] for session in sessions]
    domain_labels = [session['domain_label'] for session in sessions]
    
    pre_system_utts_list = [session['pre_system_utts'] for session in sessions]
    system_object_ids_list = [session['system_object'] for session in sessions]    
    object_datas_list = [session['object'] for session in sessions]
    backgrounds = [session['background'] for session in sessions]
    
    batch_dialogue_id = [session['dialogue_id'] for session in sessions]
    batch_turn_id = [session['turn_id'] for session in sessions]    
    
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
                object_visual_list = object_data['visual']
                
                object_feature = 0
                for object_visual in object_visual_list:
                    object_feature += img2feature(object_visual, image_feature_extractor)
                object_features.append(object_feature)
        if len(object_features) > 0:
            total_object_features.append(torch.cat(object_features, 0))
        else:
            total_object_features.append(torch.tensor([0]))
    
    return batch_tokens, batch_disamb_labels, batch_domain_labels, batch_background_features, total_object_features, batch_dialogue_id, batch_turn_id

def _CalACC(model, dataloader, test_path, args):
    """
    Expected JSON format:
        [
            "dialog_id": <dialog_id>,
            "predictions": [
                {
                    "turn_id": <turn_id>,
                    "disambiguation_label": <bool>,
                }
                ...
            ]
            ...
        ]
    """
    
    model.eval()
    
    domain, background= args.domain, args.background
    obj = args.object
    
    session_list = []
    session_dict = {'dialog_id': -1, 'predictions': []}
    
    disamb_correct, domain_correct = 0, 0    
    with torch.no_grad():
        for i_batch, (input_sample) in enumerate(tqdm(dataloader, desc='evaluation')):
            batch_tokens, disamb_label, domain_label, batch_background_features, batch_object_features, batch_dialogue_id, batch_turn_id = input_sample
            batch_tokens = batch_tokens.cuda() # (1, len)
            batch_background_features = batch_background_features.type('torch.FloatTensor').cuda() # [1, 3, 224, 224]
            batch_object_features = [x.type('torch.FloatTensor').cuda() for x in batch_object_features] # [[objnum, 3, 224, 224], ..., ]
            
            dialogue_id, turn_id = batch_dialogue_id[0], batch_turn_id[0]
            
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
                    
            temp = {}
            temp['turn_id'] = turn_id
            temp['disambiguation_label'] = pred_disamb
            if session_dict['dialog_id'] == dialogue_id:
                session_dict['predictions'].append(temp)
            else:
                if i_batch > 0:
                    session_list.append(session_dict)
                session_dict = {'dialog_id': dialogue_id, 'predictions': [temp]}
    session_list.append(session_dict)
    with open(test_path, 'w', encoding='utf-8') as make_file:
        json.dump(session_list, make_file, indent="\t")       
    
    acc = disamb_correct/len(dataloader)*100
    if domain:
        domain_acc = domain_correct/len(dataloader)*100
    else:
        domain_acc = 0
    
    return acc, domain_acc
        
def main():    
    """save & log path"""
    model_type = args.model_type
    current = args.current
    domain = args.domain
    background = args.background
    obj = args.object
    post = args.post
    model_path = args.model_path
    if os.path.basename(model_path) == 'model.pt':
        save_path = './results/dstc10-simmc-entry'
    else: # model_final.pt
        save_path = './results/dstc10-simmc-final-entry'
        
    print("###Save Path### ", save_path)
    print("use history utterance?: ", current)
    print("domain prediction learning?: ", domain)
    print("background image use?: ", background)
    print("object image use?: ", obj)
    
    log_path = os.path.join(save_path, 'test.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """Model Loading"""
    model = BaseModel(post).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)    
    model.eval()    
    
    """dataset Loading"""
    image_obj_path = "../res/image_obj.pickle"
    description_path = "../data/public/*"
    fashion_path = '../data/fashion_prefab_metadata_all.json'
    furniture_path = '../data/furniture_prefab_metadata_all.json'

    devtest_path = '../data/simmc2_dials_dstc10_devtest.json'
    devtest_dataset = task1_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path, current)
    devtest_loader = DataLoader(devtest_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)     
    
    """Testing"""    
    test_path = os.path.join(save_path, "dstc10-simmc-teststd-pred-subtask-1.json")    
    devtestAcc, devtestDomainACC = _CalACC(model, devtest_loader, test_path, args)
    logger.info("DevTestAcc: {}, DevTestDomainAcc: {}".format(devtestAcc, devtestDomainACC))

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--model_type", help = "base", default = 'roberta-large') # base
    parser.add_argument( "--model_path", type=str, help = "./model/model.pt or model_final.pt", default = './model/model_final.pt')
    
    parser.add_argument( "--current", type=str, help = 'only use current utt / system current / context', default = 'context') # current or sys_current        
    parser.add_argument('--domain', action='store_true', help='domain multi-task learning')
    parser.add_argument('--background', action='store_true', help='use background image features')
    parser.add_argument('--object', action='store_true', help='use object features')
    parser.add_argument('--post', action='store_true', help='post-trained model')
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()