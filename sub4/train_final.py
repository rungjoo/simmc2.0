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
from dataset import task4_loader
from utils import img2feature, CalBELU

""" generate model """
from transformers import GPT2Tokenizer
gpt_model_path = '/data/project/rw/rung/02_source/model/gpt2-large' # "gpt2-large" # 
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path)

text_tokenizer = gpt_tokenizer
endIDX, whichres, resIDX = -1, -1, -1

from transformers import DeiTFeatureExtractor
image_model_path = '/data/project/rw/rung/02_source/model/deit-base-distilled-patch16-224' # "facebook/deit-base-distilled-patch16-224" # 
image_feature_extractor = DeiTFeatureExtractor.from_pretrained(image_model_path)
    
def make_input(context, slots, response):
    input_str = context + ' [META] ' + slots + ' [RES] ' + response
    return input_str

def make_batch(sessions):
    # dict_keys(['sess_cnt', 'context', 'response', 'slot_values', 'request_slots', 'object_visual', 'visual_meta', 'background'])
    context_strs = [session['context'] for session in sessions]
    slot_values = [session['slot_values'] for session in sessions]
    request_slots = [session['request_slots'] for session in sessions]
    responses = [session['response'] for session in sessions]
    
    batch_visuals = [session['object_visual'] for session in sessions]
    # batch_visual_metas = [session['visual_meta'] for session in sessions]
    
    # batch_backgrounds = [session['background'] for session in sessions]
    
    batch_neg_visuals = [session['neg_object_visual'] for session in sessions]
    # batch_neg_visual_metas = [session['neg_visual_meta'] for session in sessions]
    
    """ input text tokens """
    input_strs = []
    for context, slots, response in zip(context_strs, slot_values, responses):
        input_str = make_input(context, slots, response)
        input_strs.append(input_str)
    
    batch_tokens = text_tokenizer(input_strs, padding='longest', add_special_tokens=False).input_ids # (batch, text_len, 1024)
    batch_token_list = []
    batch_res_START = []
    batch_label_tokens = []
    for batch_token in batch_tokens:
        batch_token = batch_token[-text_tokenizer.model_max_length:]
        res_st = batch_token.index(resIDX)
        batch_res_START.append(res_st)
        batch_label_tokens.append(torch.tensor(batch_token[res_st+1:]+[text_tokenizer.eos_token_id]))
        batch_token_list.append(torch.tensor(batch_token).unsqueeze(0))
    
    """ object information """
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
            
    """ negative object information"""
    batch_neg_obj_features = []
    
    for visual_list in batch_neg_visuals:
        object_visuals = []
        if len(visual_list) > 0: # visual_list: [[obj1, obj1], [obj2], [obj3]]
            for obj_list in visual_list:
                obj_feature = 0
                for visual in obj_list:
                    obj_feature += img2feature(visual, image_feature_extractor)
                object_visuals.append(obj_feature)
            batch_neg_obj_features.append(torch.cat(object_visuals,0))
        else:
            batch_neg_obj_features.append(False)
    
    return batch_token_list, batch_obj_features, batch_label_tokens, batch_res_START, batch_neg_obj_features

def make_input_generate(context, slots):
    input_str = context + ' [META] ' + slots + ' [RES]'
    return input_str

def make_batch_generate(sessions):
    # dict_keys(['sess_cnt', 'context', 'response', 'slot_values', 'request_slots', 'object_visual', 'visual_meta', 'background'])
    context_strs = [session['context'] for session in sessions]
    slot_values = [session['slot_values'] for session in sessions]
    request_slots = [session['request_slots'] for session in sessions]
    responses = [session['response'] for session in sessions]
    
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
    
    return batch_token_list, batch_obj_features, responses

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
    
    save_path = './results/dstc10-simmc-final-entry'
    
    print("###Save Path### ", save_path)
    print("post? (meta matching): ", post)
    print("object visual information?: ", obj)
    print("user_train: ", user_train)
    
    log_path = os.path.join(save_path, 'train.log')
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
        
    model = BaseModel(post).cuda()
    if args.distributed:
        model = DDP(model, delay_allreduce=True)
    model.train()    
    
    """dataset Loading"""
    image_obj_path = "../res/image_obj.pickle"
    description_path = "../data/public/*scene*"
    fashion_path = '../data/fashion_prefab_metadata_all.json'
    furniture_path = '../data/furniture_prefab_metadata_all.json'
    
    train_path = '../data/simmc2_dials_dstc10_train.json'    
    dev_path = '../data/simmc2_dials_dstc10_dev.json'
    devtest_path = '../data/simmc2_dials_dstc10_devtest.json'
            
    batch_size = args.batch
    print('batch size: ', batch_size)
    
    """ for training """
    train_dataset = task4_loader(train_path, image_obj_path, description_path, fashion_path, furniture_path, user_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=make_batch)    
    
    dev_dataset = task4_loader(dev_path, image_obj_path, description_path, fashion_path, furniture_path, user_train)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=make_batch)
    
    """ for dev """
    user_eval = False
    devtest_dataset = task4_loader(devtest_path, image_obj_path, description_path, fashion_path, furniture_path, user_eval)
    devtest_loader = DataLoader(devtest_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=make_batch_generate)
    
    """Training Parameter Setting"""    
    training_epochs = args.epoch
    print('Training Epochs: ', str(training_epochs))
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = (len(train_dataset)+len(dev_dataset))*training_epochs
    num_warmup_steps = (len(train_dataset)+len(dev_dataset))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)    
        
    """Training"""    
    logger.info('########################################')
    best_bleuscore, best_bleustd, best_epoch = -1, -1, 0    
    for epoch in range(training_epochs):
        model.train()
        for i_batch, (batch_token_list, batch_obj_features, batch_label_tokens, batch_res_START, batch_neg_obj_features) in enumerate(tqdm(train_loader, desc='train_iteration')):
            """
            batch_token_list: [(1, len), (1, len), ..., ]
            batch_obj_features: [(1, 3, 224, 224), False, ..., (1, 3, 224, 224)]
            batch_label_tokens: [(label_len), (label_len), ...]
            batch_res_START: [num, num, ...]
            batch_neg_obj_features: [(1, 3, 224, 224), False, ..., (1, 3, 224, 224)]
            """
            batch_token_list = [x.cuda() for x in batch_token_list]
            batch_label_token_list = []
            for batch_label_tokens in batch_label_tokens:
                batch_label_token_list.append(batch_label_tokens.cuda())            
            
            batch_obj_features_list = []
            for batch_obj_feature in batch_obj_features:
                if obj and type(batch_obj_feature) != type(False):
                    batch_obj_feature = batch_obj_feature.type('torch.FloatTensor').cuda()
                batch_obj_features_list.append(batch_obj_feature)
                    
            assert len(batch_token_list) == len(batch_obj_features_list)
            
            """Model Training"""
            batch_decoder_outs = model(batch_token_list, batch_obj_features_list, args)

            """ goal loss"""
            loss_val = CELoss(batch_decoder_outs, batch_label_token_list, batch_res_START)
            optimizer.zero_grad()
            loss_val.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
        for i_batch, (batch_token_list, batch_obj_features, batch_label_tokens, batch_res_START, batch_neg_obj_features) in enumerate(tqdm(dev_loader, desc='traindev_iteration')):
            """
            batch_token_list: [(1, len), (1, len), ..., ]
            batch_obj_features: [(1, 3, 224, 224), False, ..., (1, 3, 224, 224)]
            batch_label_tokens: [(label_len), (label_len), ...]
            batch_res_START: [num, num, ...]
            batch_neg_obj_features: [(1, 3, 224, 224), False, ..., (1, 3, 224, 224)]
            """
            batch_token_list = [x.cuda() for x in batch_token_list]
            batch_label_token_list = []
            for batch_label_tokens in batch_label_tokens:
                batch_label_token_list.append(batch_label_tokens.cuda())            
            
            batch_obj_features_list = []
            for batch_obj_feature in batch_obj_features:
                if obj and type(batch_obj_feature) != type(False):
                    batch_obj_feature = batch_obj_feature.type('torch.FloatTensor').cuda()
                batch_obj_features_list.append(batch_obj_feature)
                    
            assert len(batch_token_list) == len(batch_obj_features_list)
            
            """Model Training"""
            batch_decoder_outs = model(batch_token_list, batch_obj_features_list, args)

            """ goal loss"""
            loss_val = CELoss(batch_decoder_outs, batch_label_token_list, batch_res_START)
            optimizer.zero_grad()
            loss_val.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()            
            
        """ Score and Save"""        
        model.eval()
        test_bleuscore, test_bleustd = Generate(model, devtest_loader, save_path, 'test', epoch, args)
        logger.info("Test Epoch: {}, BLEU: {}, std: {}".format(epoch, test_bleuscore, test_bleustd))
        if test_bleuscore > best_bleuscore:
            _SaveModel(model, 'model')
            best_bleuscore = test_bleuscore    
            best_bleustd = test_bleustd
            best_epoch = epoch
            
    logger.info("")
    logger.info("BEST Test Epoch: {}, BLEU: {}, std: {}".format(best_epoch, best_bleuscore, best_bleustd))
    print("###Save Path### ", save_path)

def CELoss(batch_decoder_outs, batch_label_token_list, batch_res_START, ignore_index=-100):
    """
    batch_decoder_outs: [(text_len, vocab_num), ..., ]
    batch_label_token_list: [(label_len), (label_len), ...]
    batch_res_START: [batch]
    """
    loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss_val = 0
    for decoder_outs, label_tokens, res_START in zip(batch_decoder_outs, batch_label_token_list, batch_res_START):
        assert decoder_outs[-label_tokens.shape[0]:].shape[0] == label_tokens.shape[0]
        loss_val += loss(decoder_outs[-label_tokens.shape[0]:], label_tokens)
    return loss_val


def Generate(model, dataloader, save_path, dataname, epoch, args):
    obj = args.object
    
    model.eval()
    # pred_path = os.path.join(save_path, dataname+str(epoch)+'_prediction.log')
    # f = open(pred_path, 'w')
    
    ## batch = 1 in generate function
    list_predicted = []
    list_target = []
    for i_batch, (batch_token_list, batch_obj_features, responses) in enumerate(tqdm(dataloader, desc='generation')):
        """
        batch_token_list: [(1, len), (1, len), ..., ]
        batch_obj_features: [(1, 3, 224, 224), False, ..., (1, 3, 224, 224)]
        responses: [str, str, ...]
        """
        true_response = responses[0]
        
        input_tokens = batch_token_list[0].cuda()

        batch_obj_features_list = []        
        batch_obj_feature = batch_obj_features[0]
        if obj and type(batch_obj_feature) != type(False):
            batch_obj_feature = batch_obj_feature.type('torch.FloatTensor').cuda()
        batch_obj_features_list.append(batch_obj_feature)

        """Model Prediction"""
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
        # f.write(pred_response + '\t' + true_response + '\n')
    # f.close()
    
    bleuscore, bleustd = CalBELU(list_predicted, list_target)    
    return bleuscore, bleustd
            

def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model_final.pt'))   
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Subtask4" )
    parser.add_argument( "--epoch", type=int, help = 'training epochs', default = 5) # 12 for iemocap
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)    
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    parser.add_argument( "--model_type", help = "gpt2-large", default = 'gpt2-large') # large
    parser.add_argument( "--batch", type=int, help = "training batch size", default =1) 
    parser.add_argument( "--max_len", type=int, help = "generate max_len", default = 30)   
    
    parser.add_argument('--object', action='store_true', help='use object features')
    parser.add_argument('--background', action='store_true', help='use background image features')
    
    parser.add_argument('--user_train', action='store_true', help='training from system object matching')    
    
    parser.add_argument('--post', action='store_true', help='post-trained model')    
    
    parser.add_argument("--local_rank", type=int, default=0)
        
    args = parser.parse_args()    
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()