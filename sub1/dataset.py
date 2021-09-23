from torch.utils.data import Dataset, DataLoader

import random, pdb
from tqdm import tqdm

import json
import glob, os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
        
class task1_loader(Dataset):
    def __init__(self, data_path, image_obj_path, description_path, fashion_path, furniture_path, current):
        with open(data_path, 'r') as f: # './simmc2/data/simmc2_dials_dstc10_train.json'
            json_data = json.load(f)
            dialogue_data = json_data['dialogue_data']
        
        """ image input """
        try:
            with open(image_obj_path, 'rb') as f: # "../res/image_obj.pickle"
                image_visual = pickle.load(f)
        except:
            image_obj_path = os.path.splitext(image_obj_path)[0]+'_py37'+os.path.splitext(image_obj_path)[1]
            with open(image_obj_path, 'rb') as f: # "../res/image_obj.pickle"
                image_visual = pickle.load(f)
                
        image_des_list = glob.glob(description_path) # "./simmc2/data/public/*"
        
        with open(fashion_path, 'r') as f: # './simmc2/data/fashion_prefab_metadata_all.json'
            fashion_metadata = json.load(f)

        with open(furniture_path, 'r') as f: # './simmc2/data/furniture_prefab_metadata_all.json'
            furniture_metadata = json.load(f)    

        self.task1_input = {}
        self.dial2object = {}
        cnt = 0
        for dialog_cnt, one_dialogue in enumerate(dialogue_data):
            dialogue_idx, domain, mentioned_object_ids, scene_ids = one_dialogue['dialogue_idx'], one_dialogue['domain'], \
                                                                    one_dialogue['mentioned_object_ids'], one_dialogue['scene_ids']
            
            if domain == 'fashion':
                metadata = fashion_metadata
                domain_label = 0
            else:
                metadata = furniture_metadata
                domain_label = 1

            ## image information save
            self.dial2object[dialog_cnt] = {}
            self.dial2object[dialog_cnt]['object'] = {}
            self.dial2object[dialog_cnt]['background'] = []
            total_objects = []   
            
            image_data = {}            
            for k, image_name in scene_ids.items():
                if image_name[:2] == 'm_':
                    image_find_name = image_name[2:]
                else:
                    image_find_name = image_name
                image = image_visual[image_find_name]
                self.dial2object[dialog_cnt]['background'].append(image)
                
                for image_des_path in image_des_list:
                    if image_name in image_des_path: # find image file
                        with open(image_des_path, 'r') as f:
                            image_des_data = json.load(f)

                        if 'scenes' in image_des_data.keys():
                            scenes = image_des_data['scenes']
                            for scene in scenes:
                                objects, relationships = scene['objects'], scene['relationships']

                                for object_data in objects:
                                    prefab_path, unique_id, object_id, bbox, position = object_data['prefab_path'], object_data['unique_id'], object_data['index'], \
                                                    object_data['bbox'], object_data['position']
                                    total_objects.append(object_id)
                                    
                                    if object_id not in self.dial2object[dialog_cnt]['object'].keys():
                                        self.dial2object[dialog_cnt]['object'][object_id] = {}                                    

                                        left, top, height, width = bbox[0], bbox[1], bbox[2], bbox[3]
                                        object_crop = image.crop((left, top, left+width, top+height))

                                        self.dial2object[dialog_cnt]['object'][object_id]['visual'] = [object_crop]
                                        self.dial2object[dialog_cnt]['object'][object_id]['background'] = [image]
                                    else:
                                        self.dial2object[dialog_cnt]['object'][object_id]['visual'].append(object_crop)
                                        self.dial2object[dialog_cnt]['object'][object_id]['background'].append(image)
            

            """ text input """
            text_data = one_dialogue['dialogue']
            pre_system_utts = []
            system_obj_visual = []
            system_obj_ids = []

            task1_sample_input = ''
            for i, text in enumerate(text_data):
                """ user text input """
                transcript = text['transcript']        
                if i == 0:
                    task1_sample_input += '[USER] '
                else:
                    task1_sample_input += ' [USER] '
                    
                if current == 'current':
                    task1_sample_input = transcript # 진짜 curret
                elif current == 'sys_current':
                    ## system + user
                    if i == 0:
                        task1_sample_input = '[USER] ' + transcript
                    else:
                        task1_sample_input = '[SYSTEM] ' + system_transcript
                        task1_sample_input += ' [USER] ' + transcript                    
                else:
                    task1_sample_input += transcript

                if 'disambiguation_label' in text.keys():
                    """ save """
                    disambiguation_label = text['disambiguation_label']
                    self.task1_input[cnt] = {}
                    self.task1_input[cnt]['input'] = task1_sample_input                         
                    self.task1_input[cnt]['label'] = disambiguation_label
                    self.task1_input[cnt]['domain_label'] = domain_label
                    self.task1_input[cnt]['pre_system_utts'] = pre_system_utts[:]                    
                    
                    self.task1_input[cnt]['system_object'] = system_obj_ids
                    self.task1_input[cnt]['object'] = {}
                    for object_id, object_data in self.dial2object[dialog_cnt]['object'].items():
                        self.task1_input[cnt]['object'][object_id] = {}
                        self.task1_input[cnt]['object'][object_id]['visual'] = object_data['visual']
                        self.task1_input[cnt]['object'][object_id]['background'] = object_data['background']
                    self.task1_input[cnt]['background'] = self.dial2object[dialog_cnt]['background']
                    
                    cnt += 1
    
                """ system text input """
                system_transcript = text['system_transcript']
                task1_sample_input += ' [SYSTEM] '
                task1_sample_input += system_transcript
                pre_system_utts.append(system_transcript)
                
                ## system object features
                system_transcript_annotated = text['system_transcript_annotated']
                system_transcript_objects = system_transcript_annotated['act_attributes']['objects']
                                               
                for object_id in system_transcript_objects:
                    if object_id not in system_obj_ids:
                        system_obj_ids.append(object_id)

                
    def __len__(self):
        return len(self.task1_input)

    def __getitem__(self, idx):
        return self.task1_input[idx]