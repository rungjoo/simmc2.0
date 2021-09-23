from torch.utils.data import Dataset, DataLoader

import random, pdb
from tqdm import tqdm

import json
import glob, os
import pickle
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import matplotlib.pyplot as plt
        
class post_loader(Dataset):
    def __init__(self, data_path, image_obj_path, description_path, fashion_path, furniture_path):
        with open(data_path, 'r') as f: # './simmc2/data/simmc2_dials_dstc10_train.json'
            json_data = json.load(f)
            dialogue_data = json_data['dialogue_data']

        """ Image input """
        try:
            with open(image_obj_path, 'rb') as f: # "../res/image_obj.pickle"
                image_visual = pickle.load(f)
        except:
            image_obj_path = os.path.splitext(image_obj_path)[0]+'_py37'+os.path.splitext(image_obj_path)[1]
            with open(image_obj_path, 'rb') as f: # "../res/image_obj.pickle"
                image_visual = pickle.load(f)
        
        image_des_list = glob.glob(description_path) # "./simmc2/data/public/*scene*"
        
        with open(fashion_path, 'r') as f: # './simmc2/data/fashion_prefab_metadata_all.json'
            fashion_metadata = json.load(f)

        with open(furniture_path, 'r') as f: # './simmc2/data/furniture_prefab_metadata_all.json'
            furniture_metadata = json.load(f)    

        non_visual_meta_type = ['customerReview', 'brand' , 'price', 'size', 'materials']
        visual_meta_type = ['assetType', 'color' , 'pattern', 'sleeveLength', 'type']
        
        self.post_input = {}
        cnt = 0
        for dialog_cnt, one_dialogue in enumerate(dialogue_data):
            dialogue_idx, domain, mentioned_object_ids, scene_ids = one_dialogue['dialogue_idx'], one_dialogue['domain'], \
                                                                    one_dialogue['mentioned_object_ids'], one_dialogue['scene_ids']

            if domain == 'fashion':
                metadata = fashion_metadata
            else:
                metadata = furniture_metadata    
                
            """ text input """
            text_data = one_dialogue['dialogue']
            
            session_sample_input = ''
            for i, text in enumerate(text_data):
                """ user text input """
                transcript = text['transcript']        
                if i == 0:
                    session_sample_input += '[USER] '
                else:
                    session_sample_input += ' [USER] '
                session_sample_input += transcript

                transcript_annotated = text['transcript_annotated']
                transcript_objects = transcript_annotated['act_attributes']['objects']                

                """ system text input """
                system_transcript = text['system_transcript']
                session_sample_input += ' [SYSTEM] '
            session_sample_input = session_sample_input.strip()

            """ Image description save """
            total_bg = []
            for k, image_name in scene_ids.items():
                if image_name[:2] == 'm_':
                    image_find_name = image_name[2:]
                else:
                    image_find_name = image_name
                image = image_visual[image_find_name]
                
                self.post_input[cnt] = {}
                self.post_input[cnt]['text'] = session_sample_input
                self.post_input[cnt]['background'] = image
                total_bg.append(image)
                cnt += 1
                
        for key, data in self.post_input.items():
            text = data['text']
            background = data['background']
            self.post_input[key]['neg_background'] = random.choice(total_bg)
            
    def __len__(self):
        return len(self.post_input)

    def __getitem__(self, idx):
        return self.post_input[idx]    