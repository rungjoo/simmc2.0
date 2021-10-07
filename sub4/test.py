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
from test_dataset import task4_loader
from utils import img2feature, CalBELU

""" generate model """
from transformers import GPT2Tokenizer
gpt_model_path = "gpt2-large" # '/data/project/rw/rung/02_source/model/gpt2-large' # 
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path)

text_tokenizer = gpt_tokenizer
endIDX, whichres, resIDX = -1, -1, -1

from transformers import DeiTFeatureExtractor
image_model_path = "facebook/deit-base-distilled-patch16-224" # '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224' # 
image_feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)

def make_input_generate(context, slots):
    input_str = context + ' [META] ' + slots + ' [RES]'
    return input_str

def make_batch_generate(sessions):
    dialogue_idxs = [session['dialogue_idx'] for session in sessions]
    turn_idxs = [session['turn_idx'] for session in sessions]
    
    context_strs = [session['context'] for session in sessions]
    slot_values = [session['slot_values'] for session in sessions]
    request_slots = [session['request_slots'] for session in sessions]
    
    batch_visuals = [session['object_visual'] for session in sessions]
    
    # batch_backgrounds = [session['background'] for session in sessions]
    
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
    for visual_list in batch_visuals:
        object_visuals = []
        if len(visual_list) > 0: # visual_list: [[obj1, obj1], [obj2], [obj3]]
            for obj_list in visual_list: # obj_list: [obj1, obj1]
                obj_feature = 0
                for visual in obj_list:
                    obj_feature += img2feature(visual, image_feature_extractor)
                object_visuals.append(obj_feature)
            batch_obj_features.append(torch.cat(object_visuals,0))
        else:
            batch_obj_features.append(False)
    
    return batch_token_list, batch_obj_features, dialogue_idxs, turn_idxs

def main():
    """save & log path"""
    model_type = args.model_type
    obj = args.object
    post = args.post
    user_train = args.user_train

    if args.final:
        save_path = './results/dstc10-simmc-final-entry'
        model_path = './model/model_final.pt'
    else:
        save_path = './results/dstc10-simmc-entry'
        model_path = './model/model.pt'    
    
    print("###Save Path### ", save_path)
    print("post? (meta matching): ", post)
    print("object visual information?: ", obj)
    print("user_train: ", user_train)

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
        
    model = BaseModel(post).cuda()
    if args.distributed:
        model = DDP(model, delay_allreduce=True)    
    checkpoint = torch.load(model_path)
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
    else:
        image_obj_path = "../res/image_obj.pickle"
        description_path = "../data/public/*"
        devtest_path = '../data/simmc2_dials_dstc10_devtest.json'
            
    devtest_dataset = task4_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path)
    devtest_loader = DataLoader(devtest_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch_generate)
            
    """Testing"""    
    test_path = os.path.join(save_path, "dstc10-simmc-teststd-pred-subtask-4-generation.json")
    Generate(model, devtest_loader, test_path, args)

def Generate(model, dataloader, test_path, args):
    obj = args.object
    
    model.eval()
    
    ## batch = 1 in generate function    
    session_list = []
    session_dict = {'dialog_id': -1, 'predictions': []}
    
    with torch.no_grad():
        for i_batch, (batch_token_list, batch_obj_features, dialogue_idxs, turn_idxs) in enumerate(tqdm(dataloader, desc='generation')):
            """
            batch_token_list: [(1, len), (1, len), ..., ]
            batch_obj_features: [(1, 3, 224, 224), False, ..., (1, 3, 224, 224)]
            """
            dialogue_idx = dialogue_idxs[0]
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
    
    return 

    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Subtask4" )
    parser.add_argument( "--model_type", help = "gpt2-large", default = 'gpt2-large') # large
    parser.add_argument( "--max_len", type=int, help = "generate max_len", default = 30)   
    
    parser.add_argument('--object', action='store_true', help='use object features')
    parser.add_argument('--background', action='store_true', help='use background image features')    
    parser.add_argument('--user_train', action='store_true', help='training from system object matching')        
    parser.add_argument('--post', action='store_true', help='post-trained model')    
    
    parser.add_argument('--final', action='store_true', help='for dstc10-simmc-final-entry')
    
    parser.add_argument("--local_rank", type=int, default=0)
        
    args = parser.parse_args()    
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()