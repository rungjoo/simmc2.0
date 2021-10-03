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
text_model_path = '/data/project/rw/rung/02_source/model/roberta-large'
model_text_tokenizer = RobertaTokenizer.from_pretrained(text_model_path)
special_token_list = ['[USER]', '[SYSTEM]']
special_tokens = {'additional_special_tokens': special_token_list}
model_text_tokenizer.add_special_tokens(special_tokens)

from transformers import DeiTFeatureExtractor
image_model_path = '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224'        
image_feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)

def make_batch(sessions):
    # dict_keys(['input', 'object_id', 'object_label', 'dial2rel', 'dial2bg', 'system_label', 'visual', 'visual_meta'])    
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
    for visual, visual_meta in zip(visuals, visual_metas):
        object_visuals.append(img2feature(visual, image_feature_extractor))
            
    batch_obj_features = torch.cat(object_visuals,0)
        
    """ backgroubd of object features """
    bg_visuals = []
    for background in batch_backgrounds:
        bg_visuals.append(img2feature(background, image_feature_extractor))
    batch_bg_visuals = torch.cat(bg_visuals, 0)            
    
    return input_strs, batch_tokens, batch_object_labels, batch_system_labels, batch_uttcat_labels, \
            batch_obj_features, object_ids, batch_pre_system_objects, batch_bg_visuals, batch_pre_system_objects_list


def CalPER(model, dataloader, args):
    model.eval()
    
    score_type, system_matching, utt_category = args.score, args.system_matching, args.utt_category
    background, post_back = args.background, args.post_back
    original = args.original
    
    pred_list = []
    total_label_list = []
    acc_count, total_num = 0, 0
    system_pred_list, system_label_list = [], []
    uttcat_pred_list, uttcat_label_list = [], []
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
    
    def sys2pred(visual_score, system_logits_num, system_logits_list, batch_pre_system_objects):
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

        object_id = batch_object_ids[0]
        if object_id in cand_obj_ids:
            pred = score2pred(visual_score, threshold)
        elif object_id in non_cand_obj_ids:
            pred = 0
        else:
            pred = 0 # score2pred(visual_score, threshold) # 0: original
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
                    
                    # system_logits (4,2)
                    # system_labels (5) 이처럼 labels가 더 많을 수 있음 (매우 드물게) 토큰이 길면 앞의 부분을 자를 수 있으므로
                    if (len(system_labels)>0) and (len(system_logits)>0): # system label에서 학습할 발화인지 & 첫 번째 발화는 system이 없으므로, system_logits=[]을 출력
                        system_preds = system_logits.argmax(1).tolist() # [1,0,1,0,0]
                        system_labels = system_labels[-system_logits.shape[0]:] # (pred_num), 토큰이 길면 앞의 부분을 자를 수 있으므로
                        
                        for system_pred, system_label in zip(system_preds, system_labels):
                            system_pred_list.append(system_pred)
                            system_label_list.append(system_label)
                                                 
                            
            """ predicion """
            """
            utt_category_logits: (1, 4)
            """                
            uttcat_pred = utt_category_logits.argmax(1).item()
            uttcat_label = batch_uttcat_labels.item()
            if uttcat_label != -100:
                uttcat_pred_list.append(uttcat_pred)
                uttcat_label_list.append(uttcat_label)

            """
            0 # 매칭될 object 존재가 없는 발화
            1 # 이전의 system에서 언급된 object는 없고, 새로운 object가 있는 것
            2 # 이전의 system에서 언급된 object들이 후보일 경우
            3 # 이전의 system에서 언급된 object들도 있고 새로운 object도 후보일 경우     
            """
            
            object_id = batch_object_ids[0]
            batch_pre_system_objects = batch_pre_system_objects_list[0]
            pred = sys2pred(visual_score, system_logits_num, system_logits_list, batch_pre_system_objects)     
                
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
    if utt_category:
        utt_category_type = 'utt_category_use'
    else:
        utt_category_type = 'utt_category_no_use'
    meta = args.meta
    if meta:
        meta_type = 'meta_use'
    else:
        meta_type = 'meta_no_use'
    mention_inform = args.mention_inform
    if mention_inform:
        mention_inform_type = 'mention_inform_use'
    else:
        mention_inform_type = 'mention_inform_no_use'
    system_train = args.system_train
    if system_train:
        system_train_type = 'system_train_use'
    else:
        system_train_type = 'system_train_no_use'
    system_matching = args.system_matching
    if system_matching:
        system_matching_type = 'system_matching_use'
    else:
        system_matching_type = 'system_matching_no_use'
    background = args.background
    if background:
        background_type = 'background_use'
    else:
        background_type = 'background_no_use'
    post = args.post
    if post:
        post_type = 'post_use'
    else:
        post_type = 'post_no_use'
    post_back = args.post_back
    if post_back:
        post_back_type = 'post_back_use'
    else:
        post_back_type = 'post_back_no_use'
    
    if args.final:
        save_path = os.path.join('dstc', model_type+'_models', post_type+'_'+args.post_balance, score_type, current, balance_type, mention_inform_type, \
                             utt_category_type, meta_type, system_matching_type, system_train_type, background_type, post_back_type)
    else:
        save_path = os.path.join(model_type+'_models', post_type+'_'+args.post_balance, score_type, current, balance_type, mention_inform_type, \
                             utt_category_type, meta_type, system_matching_type, system_train_type, background_type, post_back_type)

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
    print("post-traeind model?: {}, type: {}".format(post, args.post_balance))
    
    log_path = os.path.join(save_path, 'test.log')
    
    fileHandler = logging.FileHandler(log_path)    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    """Model Loading"""
    args.distributed = False
    if 'WORLD_SIZE' in os.environ: 
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        print('#Num of GPU: {}'.format(int(os.environ['WORLD_SIZE'])))
        # GPU 개수개념인 듯
        # WORD_SIZE가 os에 들어가려면 torch.distributed.launch으로 실행해야함.
        # ex) python3 -m torch.distributed.launch --nproc_per_node=2 train.py    
    
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')    
        
    model = BaseModel(post_back).cuda()
    model_text_tokenizer = model.text_tokenizer
    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    
    model_path = os.path.join(save_path, 'model.pt')    
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
    devtestf1, devtestacc, devtestf1_system, devtestacc_uttcat, dstc_test_dict = CalPER(model, devtest_loader, args)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    if args.final:
        filename = post_type+'_'+args.post_balance+'_'+mention_inform_type+'_'+utt_category_type+'_'+system_matching_type+'_'+system_train_type+'_'+background_type+'_'+post_back_type+'_nouttcat_finaldstc.txt'
    else:
        filename = post_type+'_'+args.post_balance+'_'+mention_inform_type+'_'+utt_category_type+'_'+system_matching_type+'_'+system_train_type+'_'+background_type+'_'+post_back_type+'_nouttcat_dstc.txt'
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
    logger.info("DevTestf1: {}, Acc: {}, DevTestSysf1: {}, DevTestUttcat_acc: {}".format(devtestf1, devtestacc, devtestf1_system, devtestacc_uttcat))
    
    print(save_path)    
            

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--model_type", help = "large", default = 'roberta-large') # large
    parser.add_argument( "--batch", type=int, help = "training batch size", default =1) 
    parser.add_argument( "--score", type=str, help = "cosine norm or sigmoid or concat", default = 'sigmoid') # cos or sigmoid
    
    parser.add_argument( "--balance", type=str, help = 'when if utt has true object candidates', default = 'unbalance') # balance or unblance
    parser.add_argument( "--current", type=str, help = 'only use current utt / system current / context', default = 'context') # current or sys_current
    parser.add_argument('--background', action='store_true', help='use background image features')
    
    parser.add_argument('--meta', action='store_true', help='multi-task meta matching learning?')
    parser.add_argument('--system_matching', action='store_true', help='multi-task system utterance matching')
    parser.add_argument('--utt_category', action='store_true', help='multi-task utterance category prediction')
    parser.add_argument('--original', action='store_true', help='test original (not use utt_category)')
    
    parser.add_argument('--mention_inform', action='store_true', help='use mentioned obj information when training')
    parser.add_argument('--system_train', action='store_true', help='training from system object matching')    
    
    parser.add_argument('--post', action='store_true', help='post-trained model')
    parser.add_argument( "--post_balance", type=str, help = '11 / all', default = '11') # current or sys_current
    parser.add_argument('--post_back', action='store_true', help='post-trained model at background')
    
    parser.add_argument('--final', action='store_true', help='final version for dstc')
    
    parser.add_argument("--local_rank", type=int, default=0)
        
    args = parser.parse_args()    
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()