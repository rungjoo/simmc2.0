from torch.utils.data import Dataset, DataLoader

import random, pdb
from tqdm import tqdm

import json
import glob, os
import pickle
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import matplotlib.pyplot as plt
        
class task4_loader(Dataset):
    def __init__(self, data_path, image_obj_path, description_path, fashion_path, furniture_path):
        with open(data_path, 'r') as f: # './simmc2/data/simmc2_dials_dstc10_train.json'
            json_data = json.load(f)
            dialogue_data = json_data['dialogue_data']

        """ Image Input """
        with open(image_obj_path, 'rb') as f: # "../res/image_obj.pickle"
            image_visual = pickle.load(f)
        
        image_des_list = glob.glob(description_path) # "./simmc2/data/public/*scene*"
        
        with open(fashion_path, 'r') as f: # './simmc2/data/fashion_prefab_metadata_all.json'
            fashion_metadata = json.load(f)

        with open(furniture_path, 'r') as f: # './simmc2/data/furniture_prefab_metadata_all.json'
            furniture_metadata = json.load(f)    

        non_visual_meta_type = ['customerReview', 'brand' , 'price', 'size', 'materials']
        visual_meta_type = ['assetType', 'color' , 'pattern', 'sleeveLength', 'type']
        
        self.task4_input = {}
        self.dial2object = {}
        self.dial2rel = {}
        self.dial2bg = {}
        cnt = 0
        for dialog_cnt, one_dialogue in enumerate(dialogue_data):
            dialogue_idx, domain, scene_ids = one_dialogue['dialogue_idx'], one_dialogue['domain'], one_dialogue['scene_ids']

            if domain == 'fashion':
                metadata = fashion_metadata
            else:
                metadata = furniture_metadata            

            """ Image description save """
            self.dial2object[dialog_cnt] = {}            
            self.dial2rel[dialog_cnt] = {}
            self.dial2object[dialog_cnt]['object'] = {}            
            total_objects = []
            
            for k, image_name in scene_ids.items():
                if image_name[:2] == 'm_':
                    image_find_name = image_name[2:]
                else:
                    image_find_name = image_name
                image = image_visual[image_find_name]
                
                for image_des_path in image_des_list:
                    if image_name in image_des_path: # find using image
                        with open(image_des_path, 'r') as f:
                            image_des_data = json.load(f)

                        if 'scenes' in image_des_data.keys():
                            scenes = image_des_data['scenes']
                            for scene in scenes:
                                objects, relationships = scene['objects'], scene['relationships']                                
                                self.dial2rel[dialog_cnt].update(relationships)                                

                                for object_data in objects:                                    
                                    prefab_path, unique_id, object_id, bbox, position = object_data['prefab_path'], object_data['unique_id'], object_data['index'], \
                                                    object_data['bbox'], object_data['position']
                                    total_objects.append(object_id)
                         
                                    ## object 2D & meta save
                                    if object_id not in self.dial2object[dialog_cnt]['object'].keys():
                                        self.dial2object[dialog_cnt]['object'][object_id] = {}
                                        self.dial2object[dialog_cnt]['object'][object_id]['bbox'] = [bbox]

                                        left, top, height, width = bbox[0], bbox[1], bbox[2], bbox[3]
                                        object_visual = image.crop((left, top, left+width, top+height))
                                        self.dial2object[dialog_cnt]['object'][object_id]['visual'] = [object_visual]

                                        self.dial2object[dialog_cnt]['object'][object_id]['background'] = [image]
                                    else:
                                        self.dial2object[dialog_cnt]['object'][object_id]['bbox'].append(bbox)

                                        left, top, height, width = bbox[0], bbox[1], bbox[2], bbox[3]
                                        object_visual = image.crop((left, top, left+width, top+height))
                                        self.dial2object[dialog_cnt]['object'][object_id]['visual'].append(object_visual)

                                        self.dial2object[dialog_cnt]['object'][object_id]['background'].append(image)                                    
            cand_objects = total_objects                
                                        
            """ Text input """
            text_data = one_dialogue['dialogue']
            
            system2object_list = []
            utt2object_list = {}
            task4_sample_input = ''
            for i, text in enumerate(text_data):
                turn_idx = text['turn_idx']
                transcript = text['transcript']
                
                """ user text input """                
                if i == 0:
                    task4_sample_input += '[USER] '
                else:
                    task4_sample_input += ' [USER] '
                    
                task4_sample_input += transcript
                
                """ system text input """                
                system_transcript_annotated = text['system_transcript_annotated']
                system_transcript_objects = system_transcript_annotated['act_attributes']['objects']
                
                #############################################################################
                """ the condition for prediction """
                if 'slot_values' in system_transcript_annotated['act_attributes'].keys():
                    slot_values = system_transcript_annotated['act_attributes']['slot_values']
                    flatten_slots = ''
                    for key, data in slot_values.items():
                        if type(data) == type({}):
                            for data_key, data_value in data.items():
                                flatten_slots += str(data_key) + ' ' 
                                if type(data_value) == type([]):
                                    flatten_slots += ' '.join(data_value) + ' '
                                else:
                                    flatten_slots += str(data_value) + ' '
                        else:
                            flatten_slots += str(data) + ' '
                    flatten_slots = flatten_slots.strip()     

                    request_slots = system_transcript_annotated['act_attributes']['request_slots']                
                    flatten_request_slots = ''
                    for request_slot in request_slots:
                        flatten_request_slots += request_slot + ' '
                    flatten_request_slots = flatten_request_slots.strip()                

                    self.task4_input[cnt] = {}
                    self.task4_input[cnt]['sess_cnt'] = i
                    self.task4_input[cnt]['context'] = task4_sample_input
                    self.task4_input[cnt]['slot_values'] = flatten_slots          
                    self.task4_input[cnt]['request_slots'] = flatten_request_slots
                    self.task4_input[cnt]['object_visual'] = []                    
                    self.task4_input[cnt]['background'] = []
                    self.task4_input[cnt]['dialogue_idx'] = dialogue_idx
                    self.task4_input[cnt]['turn_idx'] = turn_idx

                    for obj_cnt, (object_id, dial2object_data) in enumerate(self.dial2object[dialog_cnt]['object'].items()):
                        if object_id in system_transcript_objects:   
                            self.task4_input[cnt]['object_visual'].append(dial2object_data['visual'])
                            self.task4_input[cnt]['background'].append(dial2object_data['background'])
                    cnt += 1
                    #############################################################################
                
                if i < len(text_data) - 1:
                    system_transcript = text['system_transcript']
                    task4_sample_input += ' [SYSTEM] '
                    task4_sample_input += system_transcript
                

    def __len__(self):
        return len(self.task4_input)

    def __getitem__(self, idx):
        return self.task4_input[idx]