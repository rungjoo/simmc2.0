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
    visuals = [session['visual'] for session in sessions]
    batch_backgrounds = [session['background'] for session in sessions]
    
    batch_pre_system_objects_list = [session['pre_system_objects'] for session in sessions]
    batch_object_ids = [session['object_id'] for session in sessions]
    
    """ for previous system obj_ids """
    batch_pre_system_objects = []
    for pre_system_objects_list in batch_pre_system_objects_list:
        uniq_obj_id = set()
        for utt_system_objects in pre_system_objects_list:
            for obj_id in utt_system_objects:
                uniq_obj_id.add(obj_id)
        batch_pre_system_objects.append(list(uniq_obj_id))
    
    """ text tokens """
    batch_tokens = model_text_tokenizer(input_strs, padding='longest', add_special_tokens=False).input_ids # (batch, text_len, 1024)
    batch_token_list = []
    for batch_token in batch_tokens:
        batch_token = [model_text_tokenizer.cls_token_id] + batch_token[-model_text_tokenizer.model_max_length+1:]
        batch_token_list.append(torch.tensor(batch_token).unsqueeze(0))
    batch_tokens = torch.cat(batch_token_list, 0)
    
    
    """ object features """
    object_visuals = []
    for visual in visuals:
        object_visuals.append(img2feature(visual, image_feature_extractor))
            
    batch_obj_features = torch.cat(object_visuals,0)
        
    """ backgroubd of object features """
    bg_visuals = []
    for background in batch_backgrounds:
        bg_visuals.append(img2feature(background, image_feature_extractor))
    batch_bg_visuals = torch.cat(bg_visuals, 0)            
    
    return input_strs, batch_tokens, batch_obj_features, batch_object_ids, batch_pre_system_all_objects, batch_bg_visuals, batch_pre_system_objects_list


def Matching(model, dataloader, file_path, args):
    method = args.method
    model.eval()
    
    score_type, system_matching, utt_category = args.score, args.system_matching, args.utt_category
    background, post_back = args.background, args.post_back
    
    dstc_test_dict = {}
    cc = -1
    pre_str = ''
    
    threshold = 0.5
    def score2pred(score, threshold):
        if score >= threshold:
            pred = 1
        else:
            pred = 0
        return pred
    
    def sys2pred(visual_score, system_logits_num, system_logits_list, batch_pre_system_objects, object_id):
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
            if int(method) == 1:
                pred = score2pred(visual_score, threshold)
            else:
                pred = 1
        elif object_id in non_cand_obj_ids:
            pred = 0
        else:
            pred = 0
        return pred
    
    meta = False    
    with torch.no_grad():
        for i_batch, (input_strs, batch_tokens, batch_obj_features, batch_object_ids, batch_pre_system_all_objects, batch_bg_visuals, batch_pre_system_objects_list) in enumerate(tqdm(dataloader, desc='evaluation')):
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
            batch_obj_features = batch_obj_features.type('torch.FloatTensor').cuda()
            batch_bg_visuals = batch_bg_visuals.type('torch.FloatTensor').cuda()
            
            batch_t2v_score, batch_m2v_score, system_logits_list, utt_category_logits = model(batch_tokens, None, batch_obj_features, batch_bg_visuals, \
                                                     score_type, meta, system_matching, utt_category, background, post_back)
            
            visual_score = batch_t2v_score.item()
            
            """ for using previous system objects """
            system_logits_num = 0
            if system_matching:
                """
                system_logits_list: [(len1,2), (len2,2)]
                """                                
                for system_logits in system_logits_list:
                    if system_logits != []:
                        system_logits_num += system_logits.shape[0]            
                            
            """ for using utterance category """
            uttcat_pred = utt_category_logits.argmax(1).item() # (1, 2)

            
            """ prediction """
            object_id = batch_object_ids[0]
            batch_pre_system_objects = batch_pre_system_objects_list[0]
            if system_matching:
                if utt_category:
                    if uttcat_pred == 0:
                        pred = 0
                    else:
                        pred = sys2pred(visual_score, system_logits_num, system_logits_list, batch_pre_system_objects, object_id)
                else:
                    pred = sys2pred(visual_score, system_logits_num, system_logits_list, batch_pre_system_objects, object_id)
            else: # not system_matching
                if utt_category:
                    if uttcat_pred == 0:
                        pred = 0
                    else: 
                        pred = score2pred(visual_score, threshold)
                else:
                    pred = score2pred(visual_score, threshold)
                
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
    score_type = args.score
    current = args.current
    meta = args.meta
    utt_category = args.utt_category
    system_train = args.system_train
    system_matching = args.system_matching
    background = args.background
    post = args.post
    post_back = args.post_back
    
    if args.final:
        save_path = './results/dstc10-simmc-final-entry'
        model_path = './model/model.pt'
        # model_path = './model/model_final.pt' # lack of learning time for challenge
    else:
        save_path = './results/dstc10-simmc-entry'
        model_path = './model/model.pt'

    print("###Save Path### ", save_path)
    print("score method: ", score_type)
    print("use history utterance?: ", current)
    print("multi-task (visual) meta matching learning?: ", meta)
    print("multi-task utterance category prediction?: ", utt_category)
    print("multi-task system utterance matching?", system_matching)
    print("training from system object matching?", system_train)
    print("use background image features?", background)
    print("post-traeind model?: {}, type: {}".format(post, args.post_balance))       
    
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
    else:
        image_obj_path = "../res/image_obj.pickle"
        description_path = "../data/public/*"
        devtest_path = '../data/simmc2_dials_dstc10_devtest.json' 
            
    devtest_dataset = task2_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path)
    devtest_loader = DataLoader(devtest_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch)
        
    """Testing"""
    print("Data Num ## ", len(devtest_loader))
            
    """ Prediction """
    filename = "dstc10-simmc-teststd-pred-subtask-3_"+str(args.method)+".txt"
    file_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    Matching(model, devtest_loader, file_path, args)
            

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--model_type", help = "large", default = 'roberta-large') # large
    parser.add_argument( "--score", type=str, help = "cosine norm or sigmoid or concat", default = 'sigmoid') # cos or sigmoid
    
    parser.add_argument( "--current", type=str, help = 'only use current utt / system current / context', default = 'context') # current or sys_current
    parser.add_argument('--background', action='store_true', help='use background image features')
    
    parser.add_argument('--meta', action='store_true', help='multi-task meta matching learning?')
    parser.add_argument('--system_matching', action='store_true', help='multi-task system utterance matching')
    parser.add_argument('--utt_category', action='store_true', help='multi-task utterance category prediction')
    parser.add_argument('--system_train', action='store_true', help='training from system object matching')    
    
    parser.add_argument('--post', action='store_true', help='post-trained model')
    parser.add_argument( "--post_balance", type=str, help = '11 / all', default = '11') # current or sys_current
    parser.add_argument('--post_back', action='store_true', help='post-trained model at background')
    
    parser.add_argument( "--method", type=str, help = '1 / 2', default = '1') # test method 1 or 2
    parser.add_argument('--final', action='store_true', help='final version for dstc')
    
    parser.add_argument("--local_rank", type=int, default=0)
        
    args = parser.parse_args()    
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()