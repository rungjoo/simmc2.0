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

from transformers import RobertaTokenizer
text_model_path = '/data/project/rw/rung/02_source/model/roberta-large'
model_text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
special_token_list = ['[USER]', '[SYSTEM]']
special_tokens = {'additional_special_tokens': special_token_list}
model_text_tokenizer.add_special_tokens(special_tokens)

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
    
    system_labels = [session['system_label'] for session in sessions]
    batch_pre_system_objects_list = [session['pre_system_objects'] for session in sessions]
    
    """ for utt category """    
    object_ids = [session['object_id'] for session in sessions]
    
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
    
    return input_strs, batch_tokens, batch_object_labels, batch_system_labels, object_ids, batch_pre_system_objects_list

def CalPER(model, dataloader, args):
    model.eval()
    system_matching = args.system_matching
    
    pred_list = []
    total_label_list = []
    acc_count, total_num = 0, 0
    system_pred_list, system_label_list = [], []
    dstc_test_dict = {}
    cc = -1
    pre_str = ''        
    
    def sys2pred(system_logits_num, system_logits_list, batch_pre_system_objects):
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
            pred = 1
        elif object_id in non_cand_obj_ids:
            pred = 0
        else:
            pred = 0
        return pred
    
    meta = False
    threshold = 0.5
    with torch.no_grad():
        for i_batch, (input_strs, batch_tokens, batch_object_labels, batch_system_labels, batch_object_ids, batch_pre_system_objects_list) in enumerate(tqdm(dataloader, desc='evaluation')):
            """
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
            
            system_logits_list= model(batch_tokens)
            
            label = batch_object_labels.item()
                        
            """ for system f1 """
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
            object_id = batch_object_ids[0]
            batch_pre_system_objects = batch_pre_system_objects_list[0]
            pred = sys2pred(system_logits_num, system_logits_list, batch_pre_system_objects)
                
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
    _, _, f1_system, _ = precision_recall_fscore_support(system_label_list, system_pred_list, labels=[1], average='weighted')
    
    """ acc """
    accuracy = acc_count/total_num*100
    
    return f1*100, accuracy, f1_system*100, dstc_test_dict

def main():
    """save & log path"""
    model_type = args.model_type
    balance_type = args.balance
    current = args.current
    mention_inform = args.mention_inform
    system_train = args.system_train
    system_matching = args.system_matching
    
    save_path = './results/dstc10-simmc-entry'
    
    print("###Save Path### ", save_path)
    print("use history utterance?: ", current)
    print("balance type: ", balance_type)    
    print("system utterance matching?", system_matching)
    print("use mentioned obj information when training?: ", mention_inform)
    print("training from system object matching?", system_train)
    
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
        
    model = BaseModel().cuda()
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
        for i_batch, (_, batch_tokens, batch_object_labels, batch_system_labels,  _, _) in enumerate(tqdm(train_loader, desc='iteration')):
            batch_tokens = batch_tokens.cuda()
            
            """ Model forward """            
            system_logits_list = model(batch_tokens)            
            
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

            loss_val = sysloss_val
            
            if loss_val == 0:
                continue            
                
            optimizer.zero_grad()
            loss_val.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            
        """ Score and Save"""        
        model.eval()
        devf1, devacc, devf1_system, _ = CalPER(model, dev_loader, args)
        logger.info("Epoch: {}, Devf1: {}, Acc: {}, DevSysf1: {}".format(epoch, devf1, devacc, devf1_system))
        if devf1 > best_dev_f1:
            _SaveModel(model, 'model')
            best_dev_f1 = devf1
            
            best_epoch = epoch
            devtestf1, devtestacc, devtestf1_system, dstc_test_dict = CalPER(model, devtest_loader, args)
            
            logger.info("Epoch: {}, DevTestf1: {}, Acc: {}, DevTestSysf1: {}".format(epoch, devtestf1, devtestacc, devtestf1_system))
    logger.info("")
    logger.info("BEST Devf1: {}".format(best_dev_f1))
    logger.info("BEST Epoch: {}, DevTestf1: {}, Acc: {}, DevTestSysf1: {}".format(best_epoch, devtestf1, devtestacc, devtestf1_system))
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
    
    parser.add_argument( "--balance", type=str, help = 'when if utt has true object candidates', default = 'unbalance') # balance or unblance
    parser.add_argument( "--current", type=str, help = 'only use current utt / system current / context', default = 'context') # current or sys_current
    
    parser.add_argument('--system_matching', action='store_true', help='multi-task system utterance matching')
    
    parser.add_argument('--mention_inform', action='store_true', help='use mentioned obj information when training')
    parser.add_argument('--system_train', action='store_true', help='training from system object matching')    
    
    
    parser.add_argument("--local_rank", type=int, default=0)
        
    args = parser.parse_args()    
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()