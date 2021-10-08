# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from apex.parallel import DistributedDataParallel as DDP

from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

from transformers import get_linear_schedule_with_warmup

import pdb
import argparse, logging
import glob

from model import BaseModel
# from dataset import task2_loader
from test_dataset import task2_loader

from transformers import RobertaTokenizer
text_model_path = "roberta-large" # '/data/project/rw/rung/02_source/model/roberta-large' # 
model_text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
special_token_list = ['[USER]', '[SYSTEM]']
special_tokens = {'additional_special_tokens': special_token_list}
model_text_tokenizer.add_special_tokens(special_tokens)

def make_batch(sessions):
    input_strs = [session['input'] for session in sessions]
    
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
    
    return input_strs, batch_tokens, object_ids, batch_pre_system_objects_list


def Matching(model, dataloader, file_path, args):
    model.eval()    
    
    dstc_test_dict = {}
    cc = -1
    pre_str = ''
    
    def sys2pred(system_logits_num, system_logits_list, batch_pre_system_objects, object_id):
        cand_obj_ids = []
        non_cand_obj_ids = []
        if system_logits_num > 0: # system objects들이 있는 경우
            system_logits = system_logits_list[0] # 테스트에서 batch는 , (system_utt_num, 2)
            system_cand_list = system_logits.argmax(1).tolist() # [0, 1, 1, 0]
            for system_pred, system_obj_ids in zip(system_cand_list, batch_pre_system_objects):
                if system_pred == 1:
                    cand_obj_ids += system_obj_ids
                else:
                    non_cand_obj_ids += system_obj_ids
        cand_obj_ids = list(set(cand_obj_ids))
        non_cand_obj_ids = list(set(non_cand_obj_ids))

        if object_id in cand_obj_ids:
            pred = 1
        elif object_id in non_cand_obj_ids:
            pred = 0
        else:
            pred = 0
        return pred
    
    with torch.no_grad():
        for i_batch, (input_strs, batch_tokens, batch_object_ids, batch_pre_system_objects_list) in enumerate(tqdm(dataloader, desc='evaluation')):
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
            
            system_logits_list = model(batch_tokens)
            
            """ for using previous system objects """
            system_logits_num = 0
            """
            system_logits_list: [(len1,2), (len2,2)]
            """                                
            for system_logits in system_logits_list:
                if system_logits != []:
                    system_logits_num += system_logits.shape[0]
            
            """ prediction """
            object_id = batch_object_ids[0]
            batch_pre_system_objects = batch_pre_system_objects_list[0]
            pred = sys2pred(system_logits_num, system_logits_list, batch_pre_system_objects, object_id)
                
            """ for dstc format """
            if pred == 1:
                dstc_test_dict[cc]['true_object_ids'].append(str(object_id))   
                
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
    
    return 

def main():
    """save & log path"""
    model_type = args.model_type
    current = args.current
    system_matching = args.system_matching
    
    if args.final:
        save_path = './results/dstc10-simmc-final-entry'
        model_path = './model/model.pt'
        # model_path = './model/model_final.pt' # lack of learning time for challenge
    else:
        save_path = './results/dstc10-simmc-entry'
        model_path = './model/model.pt'

    print("###Save Path### ", save_path)
    print("use history utterance?: ", current)
    print("system utterance matching?", system_matching)
    
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
    model_text_tokenizer = model.text_tokenizer
    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    
#     checkpoint = torch.load(model_path)
    checkpoint = torch.load(model_path, map_location='cuda:0')
    modify_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        if 'module' in k:
            k = k.replace('module.', '')
        modify_checkpoint[k] = v
    model.load_state_dict(modify_checkpoint, strict=False)
    model.eval()
    
    """dataset Loading"""
    fashion_path = '../data/fashion_prefab_metadata_all.json'
    furniture_path = '../data/furniture_prefab_metadata_all.json'
    
    if args.final:
        image_obj_path = "../res/image_obj_final.pickle"
        description_path = "../data/simmc2_scene_jsons_dstc10_teststd/*"
        devtest_path = '../data/simmc2_dials_dstc10_teststd_public.json'
        filename = "dstc10-simmc-teststd-pred-subtask-3_2.txt"
    else:
        image_obj_path = "../res/image_obj.pickle"
        description_path = "../data/public/*"
        devtest_path = '../data/simmc2_dials_dstc10_devtest.json' 
        filename = "dstc10-simmc-devtest-pred-subtask-3_2.txt"
            
    devtest_dataset = task2_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path)
    devtest_loader = DataLoader(devtest_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
        
    """Testing"""
    print("Data Num ## ", len(devtest_loader))
            
    """ Prediction """    
    file_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    Matching(model, devtest_loader, file_path, args)
            

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Subtask2" )
    parser.add_argument( "--model_type", help = "large", default = 'roberta-large') # large    
    
    parser.add_argument( "--current", type=str, help = 'only use current utt / system current / context', default = 'context') # current or sys_current
    parser.add_argument('--system_matching', action='store_true', help='multi-task system utterance matching')
    
    parser.add_argument('--final', action='store_true', help='final version for dstc')
    
    parser.add_argument("--local_rank", type=int, default=0)
        
    args = parser.parse_args()    
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()