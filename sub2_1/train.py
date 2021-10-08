# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from apex.parallel import DistributedDataParallel as DDP

from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup

import pdb
import argparse, logging
import glob

from model import BaseModel
from dataset import task2_loader
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
    
def clsLoss(batch_t2v_score, labels):
    """
    batch_t2v_score: [batch]
    labels: [batch]
    """
    loss = nn.BCELoss()
    
    loss_val = loss(batch_t2v_score, labels)
    
    return loss_val

def CELoss(pred_logits, labels, ignore_index=-100):
    """
    pred_logits: [batch, clsNum]
    labels: [batch]
    """
    loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss_val = loss(pred_logits, labels)
    return loss_val
        
def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    
def make_batch(sessions):
    input_strs = [session['input'] for session in sessions]
    object_labels = [session['object_label'] for session in sessions]
    visuals = [session['visual'] for session in sessions]
    
    dial2rels = [session['dial2rel'] for session in sessions]
    batch_backgrounds = [session['background'] for session in sessions]
    
    system_labels = [session['system_label'] for session in sessions]
    uttcat_labels = [session['uttcat_label'] for session in sessions]
    batch_meta_labels = torch.ones([len(visuals)])
    
    batch_pre_system_objects_list = [session['pre_system_objects'] for session in sessions]
    
    """ for utt category """    
    object_ids = [session['object_id'] for session in sessions]
    batch_pre_system_all_objects = []
    for pre_system_objects_list in batch_pre_system_objects_list:
        uniq_obj_id = set()
        for utt_system_objects in pre_system_objects_list:
            for obj_id in utt_system_objects:
                uniq_obj_id.add(obj_id)
        batch_pre_system_all_objects.append(list(uniq_obj_id))
    
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
    
    """ object&meta features """
    meta_strs = []
    object_visuals = []
    for visual, visual_meta in zip(visuals, visual_metas):
        meta_strs.append(visual_meta)
        object_visuals.append(img2feature(visual, image_feature_extractor))
            
    batch_meta_tokens = model_text_tokenizer(meta_strs, padding='longest', return_tensors='pt').input_ids
    batch_obj_features = torch.cat(object_visuals,0)
    
    bg_visuals = []
    for background in batch_backgrounds:
        bg_visuals.append(img2feature(background, image_feature_extractor))
    batch_bg_visuals = torch.cat(bg_visuals, 0) 
    
    return input_strs, batch_tokens, batch_object_labels, batch_system_labels, batch_uttcat_labels, batch_meta_labels, batch_meta_tokens, \
            batch_obj_features, object_ids, batch_pre_system_all_objects, batch_bg_visuals, batch_pre_system_objects_list

def CalPER(model, dataloader, args):
    model.eval()
    
    score_type, system_matching, utt_category = args.score, args.system_matching, args.utt_category
    background, post_back = args.background, args.post_back
    
    pred_list = []
    total_label_list = []
    acc_count, total_num = 0, 0
    system_pred_list, system_label_list = [], []
    uttcat_pred_list, uttcat_label_list = [], []
    dstc_test_dict = {}
    cc = -1
    pre_str = ''    
    
    def score2pred(score, threshold):
        if score >= threshold:
            pred = 1
        else:
            pred = 0
        return pred
    
    def sys2pred(visual_score, system_logits_num, system_logits_list, batch_pre_system_objects):
        cand_obj_ids = []
        non_cand_obj_ids = []
        if system_logits_num > 0: # if system objects exist
            system_logits = system_logits_list[0] # (system_utt_num, 2) at test
            system_cand_list = system_logits.argmax(1).tolist() # [0, 1, 1, 0]
            for system_pred, system_obj_ids in zip(system_cand_list, batch_pre_system_objects):
                if system_pred == 1:
                    cand_obj_ids += system_obj_ids
                else:
                    non_cand_obj_ids += system_obj_ids
        cand_obj_ids = list(set(cand_obj_ids))
        non_cand_obj_ids = list(set(non_cand_obj_ids))

        object_id = batch_object_ids[0]
        if object_id in cand_obj_ids:
            pred = score2pred(visual_score, threshold)
        elif object_id in non_cand_obj_ids:
            pred = 0
        else:
            pred = 0 # score2pred(visual_score, threshold)
        return pred
    
    meta = False
    threshold = 0.5
    with torch.no_grad():
        for i_batch, (input_strs, batch_tokens, batch_object_labels, batch_system_labels, batch_uttcat_labels, batch_meta_labels, batch_meta_tokens, batch_obj_features, batch_object_ids, batch_pre_system_all_objects, batch_bg_visuals, batch_pre_system_objects_list) in enumerate(tqdm(dataloader, desc='evaluation')):
            """
            batch_pre_system_all_objects: unique system object ids of previous system's utterances 
            batch_pre_system_objects_list: object ids corresponding to the previous each system's utterance
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
            batch_meta_labels = batch_meta_labels
            batch_meta_tokens = batch_meta_tokens.cuda()
            batch_obj_features = batch_obj_features.type('torch.FloatTensor').cuda()
            batch_bg_visuals = batch_bg_visuals.type('torch.FloatTensor').cuda()
            
            batch_t2v_score, batch_m2v_score, system_logits_list, utt_category_logits = model(batch_tokens, batch_meta_tokens, batch_obj_features, batch_bg_visuals, \
                                                     score_type, meta, system_matching, utt_category, background, post_back)
            
            visual_score = batch_t2v_score.item()
            label = batch_object_labels.item()
                        
            """ for system f1 """
            if system_matching:
                """
                batch_system_labels: [[-100,-100,0,0,1], [0,1,0,0,1], []]
                system_logits_list: [(len1,2), (len2,2)]
                """                
                system_logits_num = 0
                for system_logits, system_labels in zip(system_logits_list, batch_system_labels):
                    if system_logits != []:
                        system_logits_num += system_logits.shape[0]
                    
                    if (len(system_labels)>0) and (len(system_logits)>0): # Only if there are labels to evaluate
                        system_preds = system_logits.argmax(1).tolist() # [1,0,1,0,0]
                        system_labels = system_labels[-system_logits.shape[0]:]
                        
                        for system_pred, system_label in zip(system_preds, system_labels):
                            system_pred_list.append(system_pred)
                            system_label_list.append(system_label)
                                                 
                            
            """ predicion """
            """
            utt_category_logits: (1, 2)
            """                
            uttcat_pred = utt_category_logits.argmax(1).item()
            uttcat_label = batch_uttcat_labels.item()
            if uttcat_label != -100:
                uttcat_pred_list.append(uttcat_pred)
                uttcat_label_list.append(uttcat_label)
            
            object_id = batch_object_ids[0]
            batch_pre_system_objects = batch_pre_system_objects_list[0]
            if system_matching:
                if utt_category:
                    if uttcat_pred == 0:
                        pred = 0
                    else:
                        pred = sys2pred(visual_score, system_logits_num, system_logits_list, batch_pre_system_objects)
                else:
                    pred = sys2pred(visual_score, system_logits_num, system_logits_list, batch_pre_system_objects)
            else: # not system_matching
                if utt_category:
                    if uttcat_pred == 0:
                        pred = 0
                    else: # 3
                        pred = score2pred(visual_score, threshold)
                else:
                    pred = score2pred(visual_score, threshold)
                
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
    if system_matching:
        _, _, f1_system, _ = precision_recall_fscore_support(system_label_list, system_pred_list, labels=[1], average='weighted')
    else:
        f1_system = 0
    if utt_category:
        utt_cnt = 0
        for utt_pred, utt_label in zip(uttcat_pred_list, uttcat_label_list):
            if utt_pred == utt_label:
                utt_cnt += 1
        acc_uttcat = utt_cnt/len(uttcat_pred_list)*100
    else:
        acc_uttcat = 0
    
    """ acc """
    accuracy = acc_count/total_num*100
    
    return f1*100, accuracy, f1_system*100, acc_uttcat, dstc_test_dict

def main():
    """save & log path"""
    model_type = args.model_type
    score_type = args.score
    balance_type = args.balance
    current = args.current
    utt_category = args.utt_category
    meta = args.meta
    mention_inform = args.mention_inform
    system_train = args.system_train
    system_matching = args.system_matching
    background = args.background
    post = args.post
    post_back = args.post_back
    
    save_path = './results/dstc10-simmc-entry'
    
    print("###Save Path### ", save_path)
    print("score method: ", score_type)
    print("use history utterance?: ", current)
    print("balance type: ", balance_type)
    print("multi-task utterance category prediction?: ", utt_category)
    print("multi-task (visual) meta matching learning?: ", meta)
    print("multi-task system utterance matching?", system_matching)
    print("use mentioned obj information when training?: ", mention_inform)
    print("training from system object matching?", system_train)
    print("use background image features?", background)
    print("post-trained model?: {}, type: {}".format(post, args.post_balance))
    
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    """Model Loading"""
    args.distributed = False
    if 'WORLD_SIZE' in os.environ: 
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        print('#Num of GPU: {}'.format(int(os.environ['WORLD_SIZE'])))
    
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')    
        
    model = BaseModel(post_back).cuda()
    if post:
        post_path = "../ITM/post_training"
        post_model = os.path.join(post_path, args.post_balance, 'model.pt')
        checkpoint = torch.load(post_model)
        model.load_state_dict(checkpoint, strict=False)
        print('Post-trained Model Loading!!')
    if args.distributed:
        model = DDP(model, delay_allreduce=True)
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
    
    train_dataset = task2_loader(train_path, image_obj_path, description_path, fashion_path, furniture_path, current, balance_type, mention_inform, system_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=make_batch)
    
    mention_eval = False
    system_eval = False
    dev_dataset = task2_loader(dev_path, image_obj_path, description_path, fashion_path, furniture_path, current, 'unbalance', mention_eval, system_eval)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
    
    devtest_dataset = task2_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path, current, 'unbalance', mention_eval, system_eval)
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
    best_dev_f1, best_epoch = -1, 0
    print("Data Num ## ", len(train_loader))
    for epoch in range(training_epochs):
        model.train()
        for i_batch, (_, batch_tokens, batch_object_labels, batch_system_labels, batch_uttcat_labels, batch_meta_labels, batch_meta_tokens, batch_obj_features, _, _, batch_bg_visuals, _) in enumerate(tqdm(train_loader, desc='iteration')):            
            batch_tokens = batch_tokens.cuda()
            batch_object_labels = batch_object_labels.type('torch.FloatTensor').cuda()
            # batch_system_labels = batch_system_labels.cuda()
            batch_uttcat_labels = batch_uttcat_labels.cuda()
            batch_meta_labels = batch_meta_labels.type('torch.FloatTensor').cuda()
            batch_meta_tokens = batch_meta_tokens.cuda()
            batch_obj_features = batch_obj_features.type('torch.FloatTensor').cuda()
            batch_bg_visuals = batch_bg_visuals.type('torch.FloatTensor').cuda()
            
            """ Model forward """            
            batch_t2v_score, batch_m2v_score, system_logits_list, utt_category_logits = model(batch_tokens, batch_meta_tokens, batch_obj_features, batch_bg_visuals, \
                                                     score_type, meta, system_matching, utt_category, background, post_back)
            
            """ goal loss"""
            clsloss_val = clsLoss(batch_t2v_score, batch_object_labels)

            """ multi-task loss"""
            if meta:
                metaloss_val = clsLoss(batch_m2v_score, batch_meta_labels)
            else:
                metaloss_val = 0
            if system_matching:
                """
                batch_system_labels: [[-100,-100,0,0,1], [0,1,0,0,1], []]
                tensor([[-100,-100,0,0,1], [0,1,0,0,1]]) (2,5)
                system_logits_list: [(len1,2), (len2,2)]
                """
                sysloss_val = 0
                for system_logits, system_labels in zip(system_logits_list, batch_system_labels):                    
                    """ 
                        Example)
                        system_logits (4,2)
                        system_labels (5) There can be more labels like this (very rarely) because long tokens can cut the front part
                        When learning system utterances, there is an empty list [] in system_labels, so no training
                        Even if it learns with user utterances, there is no system for utterances in the first turn, so system_logits=[] is output.               
                    """
                    if (len(system_labels)>0) and (len(system_logits)>0): 
                        system_labels = system_labels[-system_logits.shape[0]:]
                        system_labels = torch.tensor(system_labels).cuda()
                        sysloss_val += CELoss(system_logits, system_labels)
            else:
                sysloss_val = 0
            if utt_category:                
                uttloss_val = CELoss(utt_category_logits, batch_uttcat_labels)
            else:
                uttloss_val = 0

            loss_val = clsloss_val + metaloss_val + sysloss_val + uttloss_val
                
            optimizer.zero_grad()
            loss_val.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            
        """ Score and Save"""        
        model.eval()
        devf1, devacc, devf1_system, devacc_uttcat, _ = CalPER(model, dev_loader, args)
        logger.info("Epoch: {}, Devf1: {}, Acc: {}, DevSysf1: {}, DevUttcat_acc: {}".format(epoch, devf1, devacc, devf1_system, devacc_uttcat))
        if devf1 > best_dev_f1:
            _SaveModel(model, 'model')
            best_dev_f1 = devf1
            
            best_epoch = epoch
            devtestf1, devtestacc, devtestf1_system, devtestacc_uttcat, dstc_test_dict = CalPER(model, devtest_loader, args)
            
            logger.info("Epoch: {}, DevTestf1: {}, Acc: {}, DevTestSysf1: {}, DevTestUttcat_acc: {}".format(epoch, devtestf1, devtestacc, devtestf1_system, devtestacc_uttcat))
    logger.info("")
    logger.info("BEST Devf1: {}".format(best_dev_f1))
    logger.info("BEST Epoch: {}, DevTestf1: {}, Acc: {}, DevTestSysf1: {}, DevTestUttcat_acc: {}".format(best_epoch, devtestf1, devtestacc, devtestf1_system, devtestacc_uttcat))
    print("###Save Path### ", save_path)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Subtask2" )
    parser.add_argument( "--epoch", type=int, help = 'training epochs', default = 5) 
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)    
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    parser.add_argument( "--model_type", help = "large", default = 'roberta-large') # large
    parser.add_argument( "--batch", type=int, help = "training batch size", default =1) 
    parser.add_argument( "--score", type=str, help = "cosine norm or sigmoid or concat", default = 'sigmoid') # cos or sigmoid
    
    parser.add_argument( "--balance", type=str, help = 'when if utt has true object candidates', default = 'unbalance') # balance or unblance
    parser.add_argument( "--current", type=str, help = 'only use current utt / system current / context', default = 'context') # current or sys_current
    parser.add_argument('--relation', action='store_true', help='use around object features')
    parser.add_argument('--background', action='store_true', help='use background image features')
    
    parser.add_argument('--meta', action='store_true', help='multi-task meta matching learning?')
    parser.add_argument('--system_matching', action='store_true', help='multi-task system utterance matching')
    parser.add_argument('--utt_category', action='store_true', help='multi-task utterance category prediction')
    
    parser.add_argument('--mention_inform', action='store_true', help='use mentioned obj information when training')
    parser.add_argument('--system_train', action='store_true', help='training from system object matching')    
    
    parser.add_argument('--post', action='store_true', help='post-trained model')
    parser.add_argument( "--post_balance", type=str, help = '11 / all', default = '11') # balance at ITM
    parser.add_argument('--post_back', action='store_true', help='post-trained model at background') # if post_back is not, the model used for object and background is shared.
    
    parser.add_argument("--local_rank", type=int, default=0)
        
    args = parser.parse_args()    
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()