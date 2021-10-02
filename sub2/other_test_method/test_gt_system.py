# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from torch.utils.data import Dataset, DataLoader

import pdb
import argparse, logging
import glob

from model import BaseModel
from test_dataset import task2_loader
from utils import img2feature

from transformers import RobertaTokenizer
text_model_path = '/data/project/rw/rung/02_source/model/roberta-large'
model_text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
special_token_list = ['[USER]', '[SYSTEM]']
special_tokens = {'additional_special_tokens': special_token_list}
model_text_tokenizer.add_special_tokens(special_tokens)

from transformers import DeiTFeatureExtractor
image_model_path = '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224'        
image_feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)

def make_batch(sessions): 
    input_strs = [session['input'] for session in sessions]
    object_labels = [session['object_label'] for session in sessions]
    visuals = [session['visual'] for session in sessions]
        
    batch_backgrounds = [session['background'] for session in sessions]
    
    system_labels = [session['system_label'] for session in sessions]
    uttcat_labels = [session['uttcat_label'] for session in sessions]
    
    batch_pre_system_objects_list = [session['pre_system_objects'] for session in sessions]
    
    """ for utt category """    
    object_ids = [session['object_id'] for session in sessions]
    batch_pre_system_objects = []
    for pre_system_objects_list in batch_pre_system_objects_list:
        uniq_obj_id = set()
        for utt_system_objects in pre_system_objects_list:
            for obj_id in utt_system_objects:
                uniq_obj_id.add(obj_id)
        batch_pre_system_objects.append(list(uniq_obj_id))
    
    visual_metas = [session['visual_meta'] for session in sessions]
    
    """ text tokens """
    batch_tokens = model_text_tokenizer(input_strs, padding='longest', add_special_tokens=False).input_ids # (batch, text_len, 1024)
    batch_token_list = []
    for batch_token in batch_tokens:
        batch_token = [model_text_tokenizer.cls_token_id] + batch_token[-model_text_tokenizer.model_max_length+1:]
        batch_token_list.append(torch.tensor(batch_token).unsqueeze(0))
    batch_tokens = torch.cat(batch_token_list, 0)
    
    """ labels """
    batch_object_labels = torch.tensor(object_labels)
    batch_system_labels = system_labels
    batch_uttcat_labels = torch.tensor(uttcat_labels)
    
    """ object features """
    object_visuals = []
    for visual_list in visuals:
        visual_features = 0
        for visual in visual_list:
            visual_features += img2feature(visual, image_feature_extractor)
        object_visuals.append(visual_features)        
            
    batch_obj_features = torch.cat(object_visuals,0)
    
    """ backgroubd of object features """
    bg_visuals = []
    for background_list in batch_backgrounds:
        background_features = 0
        for background in background_list:
            background_features += img2feature(background, image_feature_extractor)
        bg_visuals.append(background_features)
    batch_bg_visuals = torch.cat(bg_visuals, 0)        
    
    return input_strs, batch_tokens, batch_object_labels, batch_system_labels, batch_uttcat_labels, \
            batch_obj_features, object_ids, batch_pre_system_objects, batch_bg_visuals, batch_pre_system_objects_list


def CalPER(dataloader):
    pred_list = []
    total_label_list = []
    acc_count, total_num = 0, 0
    system_pred_list, system_label_list = [], []
    uttcat_pred_list, uttcat_label_list = [], []
    dstc_test_dict = {}
    cc = -1
    pre_str = ''    
    
    def sys2pred(batch_pre_system_objects, label):
        cand_obj_ids = []
        for system_obj_ids in batch_pre_system_objects:
            cand_obj_ids += system_obj_ids
        cand_obj_ids = list(set(cand_obj_ids))

        object_id = batch_object_ids[0]
        if (object_id in cand_obj_ids): # and (label == 1):
            pred = 1
        else:
            pred = 0
        return pred    
    
    meta = False    
    with torch.no_grad():
        for i_batch, (input_strs, batch_tokens, batch_object_labels, batch_system_labels, batch_uttcat_labels, batch_obj_features, batch_object_ids, batch_pre_system_objects, batch_bg_visuals, batch_pre_system_objects_list) in enumerate(tqdm(dataloader, desc='evaluation')):
            """
            batch_pre_system_objects: 이전의 unique system object ids
            batch_pre_system_objects_list: 이전의 각 시스템 발화에 해당하는 object ids
            """
            input_str = input_strs[0]
            if input_str != pre_str:
                cc += 1
                dstc_test_dict[cc] = {}
                dstc_test_dict[cc]['input_str'] = input_str
                dstc_test_dict[cc]['true_object_ids'] = []
            pre_str = input_str
            
            batch_tokens = batch_tokens.cuda()
            batch_object_labels = batch_object_labels
            batch_system_labels = batch_system_labels
            batch_meta_tokens = 0
            batch_obj_features = batch_obj_features.type('torch.FloatTensor').cuda()
            batch_bg_visuals = batch_bg_visuals.type('torch.FloatTensor').cuda()
            
            label = batch_object_labels.item()
                        
            object_id = batch_object_ids[0]
            batch_pre_system_objects = batch_pre_system_objects_list[0]
            pred = sys2pred(batch_pre_system_objects, label)
                
            """ for F1 """
            pred_list.append(pred)
            total_label_list.append(label)

            """ for acc """
            total_num += 1
            if pred == label:
                acc_count += 1
                
            """ for dstc """
            if pred == 1:
                dstc_test_dict[cc]['true_object_ids'].append(str(object_id))
                    
    """ preicions & recall & f1"""
    precision, recall, f1, _ = precision_recall_fscore_support(total_label_list, pred_list, labels=[1], average='weighted')
    
    """ acc """
    accuracy = acc_count/total_num*100
    
    return f1*100, accuracy, dstc_test_dict

def main():
    """save & log path"""
    save_path = 'results'
    
    log_path = os.path.join(save_path, 'test_gt.log')
    
    fileHandler = logging.FileHandler(log_path)    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """dataset Loading"""
    image_obj_path = "/data/project/rw/rung/00_company/03_DSTC10_SIMMC/res/image_obj.pickle"
    description_path = "/data/project/rw/rung/00_company/03_DSTC10_SIMMC/simmc2/data/public/*scene*"
    fashion_path = '/data/project/rw/rung/00_company/03_DSTC10_SIMMC/simmc2/data/fashion_prefab_metadata_all.json'
    furniture_path = '/data/project/rw/rung/00_company/03_DSTC10_SIMMC/simmc2/data/furniture_prefab_metadata_all.json'
    
    devtest_path = '/data/project/rw/rung/00_company/03_DSTC10_SIMMC/simmc2/data/simmc2_dials_dstc10_devtest.json'
            
    devtest_dataset = task2_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path)
    devtest_loader = DataLoader(devtest_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
        
    """Testing"""
    print("Data Num ## ", len(devtest_loader))
            
    """ Score """
    devtestf1, devtestacc, dstc_test_dict = CalPER(devtest_loader)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    filename = 'gt_system_alltrue.txt'
    file_path = os.path.join('results', filename)    
    f = open(file_path, 'w')    
    for session, data in dstc_test_dict.items():
        input_str, object_list = data['input_str'], data['true_object_ids']
        object_list = list(set(object_list))
        input_str = input_str.replace('[USER]', 'User :')
        input_str = input_str.replace('[SYSTEM]', 'System :')
        input_str += " => Belief State :"
        input_str += " REQUEST:GET [ type = blouse ] (availableSizes, pattern) < "
        input_str += ', '.join(object_list)
        input_str += " > <EOB> DUMP <EOS>"
        f.write(input_str+'\n')
    f.close()
    logger.info("DevTestf1: {}, Acc: {}".format(devtestf1, devtestacc))
    
    print(save_path)    
            

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()