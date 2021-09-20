# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from apex.parallel import DistributedDataParallel as DDP
from collections import OrderedDict

from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup

import pdb, json
import argparse, logging
import glob

from model import BaseModel
from dataset import task4_loader
from utils import img2feature, CalBELU

""" generate model """
from transformers import GPT2Tokenizer
gpt_model_path = '/data/project/rw/rung/02_source/model/gpt2-large'
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path)

text_tokenizer = gpt_tokenizer
endIDX, whichres, resIDX = -1, -1, -1

from transformers import DeiTFeatureExtractor
image_model_path = '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224'        
image_feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)

def make_input_generate(context, slots):
    input_str = context + ' [META] ' + slots + ' [RES]'
    return input_str

def make_batch_generate(sessions):
    # dict_keys(['sess_cnt', 'context', 'response', 'slot_values', 'request_slots', 'object_visual', 'visual_meta', 'background', 'dialogue_idx', 'turn_idx'])
    dialogue_idxs = [session['dialogue_idx'] for session in sessions]
    turn_idxs = [session['turn_idx'] for session in sessions]
    
    context_strs = [session['context'] for session in sessions]
    slot_values = [session['slot_values'] for session in sessions]
    request_slots = [session['request_slots'] for session in sessions]
    responses = [session['response'] for session in sessions]
    
    batch_visuals = [session['object_visual'] for session in sessions]
    batch_visual_metas = [session['visual_meta'] for session in sessions]
    
    batch_backgrounds = [session['background'] for session in sessions]
    
    input_strs = []
    for context, slots in zip(context_strs, slot_values):
        input_str = make_input_generate(context, slots)
        input_strs.append(input_str)
    
    batch_tokens = text_tokenizer(input_strs, padding='longest', add_special_tokens=False).input_ids # (batch, text_len, 1024)
    batch_token_list = []
    for batch_token in batch_tokens:
        batch_token = batch_token[-text_tokenizer.model_max_length:]        
        batch_token_list.append(torch.tensor(batch_token).unsqueeze(0))
    
    batch_obj_features = []    
    for visual_list in zip(batch_visuals):
        object_visuals = []
        if len(visual_list) > 0:
            for visual in zip(visual_list):
                object_visuals.append(img2feature(visual, image_feature_extractor))
            batch_obj_features.append(torch.cat(object_visuals,0))
        else:
            batch_obj_features.append(False)
    
    return batch_token_list, batch_obj_features, responses, dialogue_idxs, turn_idxs

def main():
    """save & log path"""
    model_type = args.model_type
    obj = args.object
    if obj:
        obj_type = 'object_use'
    else:
        obj_type = 'object_no_use'    
    post = args.post
    if post:
        post_type = 'post_use'
    else:
        post_type = 'post_no_use' 
    user_train = args.user_train
    if user_train:
        user_train_type = 'user_train_use'
    else:
        user_train_type = 'user_train_no_use'
    meta = args.meta
    
    # save_path = os.path.join(model_type+'_models', post_type, obj_type, user_train_type)
    save_path = 'results'
    
    print("###Save Path### ", save_path)
    print("post? (meta matching): ", post)
    print("object visual information?: ", obj)
    print("user_train: ", user_train)
    
    log_path = os.path.join(save_path, 'test.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)    
    
    """tokenizer"""
    global text_tokenizer, endIDX, whichres, resIDX
    text_tokenizer = gpt_tokenizer
    special_token_list = ['[USER]', '[SYSTEM]', '[RES]', '[META]']
    special_tokens = {'additional_special_tokens': special_token_list, 'pad_token': '[PAD]'}
    text_tokenizer.add_special_tokens(special_tokens)
    
    endIDX = text_tokenizer.eos_token_id
    whichres = text_tokenizer.all_special_tokens.index('[RES]')
    resIDX = text_tokenizer.all_special_ids[whichres]
    
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
        
    model = BaseModel(model_type, post).cuda()
    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    model_path = os.path.join('model', 'model.pt')    
    checkpoint = torch.load(model_path)
    modify_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        if 'module' in k:
            k = k.replace('module.', '')
        modify_checkpoint[k] = v
    model.load_state_dict(modify_checkpoint, strict=False)
    model.eval()
    
    """dataset Loading"""
    image_obj_path = "../res/image_obj.pickle"
    description_path = "/data/project/rw/rung/00_company/03_DSTC10_SIMMC/simmc2/data/public/*scene*"
    fashion_path = '/data/project/rw/rung/00_company/03_DSTC10_SIMMC/simmc2/data/fashion_prefab_metadata_all.json'
    furniture_path = '/data/project/rw/rung/00_company/03_DSTC10_SIMMC/simmc2/data/furniture_prefab_metadata_all.json'
    
    devtest_path = '/data/project/rw/rung/00_company/03_DSTC10_SIMMC/simmc2/data/simmc2_dials_dstc10_devtest.json'
            
    user_eval = False
    devtest_dataset = task4_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path, user_eval)
    devtest_loader = DataLoader(devtest_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch_generate)
            
    """Testing"""   
    if not os.path.exists('results'):
        os.makedirs('results')
    test_path = os.path.join('results', "dstc10-simmc-teststd-pred-subtask-4-generation.json")    
    test_bleuscore, test_bleustd = Generate(model, devtest_loader, test_path, args)
    logger.info("BLEU: {}, std: {}".format(test_bleuscore, test_bleustd))

def Generate(model, dataloader, test_path, args):
    obj = args.object
    meta = args.meta
    
    model.eval()
    
    ## batch = 1 in generate function
    list_predicted = []    
    list_target = []
    
    session_list = []
    session_dict = {'dialog_id': -1, 'predictions': []}
    
    for i_batch, (batch_token_list, batch_obj_features, responses, dialogue_idxs, turn_idxs) in enumerate(tqdm(dataloader, desc='generation')):
        """
        batch_token_list: [(1, len), (1, len), ..., ]
        batch_obj_features: [(1, 3, 224, 224), False, ..., (1, 3, 224, 224)]
        responses: [str, str, ...]
        """
        dialogue_idx = dialogue_idxs[0]
        true_response = responses[0]        
        input_tokens = batch_token_list[0].cuda()

        batch_obj_features_list = []        
        batch_obj_feature = batch_obj_features[0]
        if obj and type(batch_obj_feature) != type(False):
            batch_obj_feature = batch_obj_feature.type('torch.FloatTensor').cuda()
        batch_obj_features_list.append(batch_obj_feature)

        """Next token prediction"""
        for _ in range(args.max_len):
            batch_decoder_out = model([input_tokens], batch_obj_features_list, args)[0] # (text_len, vocab_num)            
            max_ind = torch.argmax(batch_decoder_out[-1,:], 0)            
            if endIDX == max_ind.item():
                break
            input_tokens = torch.cat([input_tokens, max_ind.unsqueeze(0).unsqueeze(0)], 1) # (1, len)            
        out_str = text_tokenizer.decode(input_tokens.tolist()[0])
        context = out_str.split('[RES]')[0].strip()
        pred_response = out_str.split('[RES]')[-1].strip()
        list_predicted.append(pred_response)
        
        list_target.append(true_response)
        
        temp = {}
        temp['turn_id'] = turn_idxs[0]
        temp['response'] = pred_response
        if session_dict['dialog_id'] == dialogue_idxs[0]:
            session_dict['predictions'].append(temp)
        else:
            if i_batch > 0:
                session_list.append(session_dict)
            session_dict = {'dialog_id': dialogue_idxs[0], 'predictions': [temp]}
    session_list.append(session_dict)
    with open(test_path, 'w', encoding='utf-8') as make_file:
        json.dump(session_list, make_file, indent="\t")    
    
    bleuscore, bleustd = CalBELU(list_predicted, list_target)    
    return bleuscore, bleustd

    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Subtask4" )
    parser.add_argument( "--model_type", help = "gpt2-large", default = 'gpt2-large') # large
    parser.add_argument( "--max_len", type=int, help = "generate max_len", default = 30)   
    
    parser.add_argument('--object', action='store_true', help='use object features')
    parser.add_argument('--background', action='store_true', help='use background image features')
    
    parser.add_argument('--meta', action='store_true', help='multi-task meta matching learning?')
    parser.add_argument('--user_train', action='store_true', help='training from system object matching')    
    
    parser.add_argument('--post', action='store_true', help='post-trained model')    
    
    parser.add_argument("--local_rank", type=int, default=0)
        
    args = parser.parse_args()    
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()