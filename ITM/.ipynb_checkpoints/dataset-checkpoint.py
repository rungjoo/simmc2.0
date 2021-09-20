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
        self.dial2object = {}
        self.dial2rel = {}
        self.dial2bg = {}
        cnt = 0
        for dialog_cnt, one_dialogue in enumerate(dialogue_data):
            dialogue_idx, domain, mentioned_object_ids, scene_ids = one_dialogue['dialogue_idx'], one_dialogue['domain'], \
                                                                    one_dialogue['mentioned_object_ids'], one_dialogue['scene_ids']

            if domain == 'fashion':
                metadata = fashion_metadata
            else:
                metadata = furniture_metadata            

            """ Image description save """
            self.dial2object[dialog_cnt] = {}            
            self.dial2rel[dialog_cnt] = {}
            self.dial2bg[dialog_cnt] = {}
            self.dial2object[dialog_cnt] = []
            
            for k, image_name in scene_ids.items():
                if image_name[:2] == 'm_':
                    image_find_name = image_name[2:]
                else:
                    image_find_name = image_name
                image = image_visual[image_find_name]
                self.dial2bg[dialog_cnt][image_find_name] = image
                
                for image_des_path in image_des_list:
                    if image_name in image_des_path: # find image name
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

                                    ## object 2D & meta save
                                    visual_metalist = []
                                    non_visual_metalist = []
                                    for k, v in metadata[prefab_path].items():
                                        if k in visual_meta_type:
                                            visual_metalist.append(v)
                                        elif k in non_visual_meta_type:
                                            non_visual_metalist.append(v)
                                    visual_meta_flatten = ' '.join([str(x) for x in visual_metalist])
                                    non_visual_meta_flatten = ' '.join([str(x) for x in non_visual_metalist])

                                    left, top, height, width = bbox[0], bbox[1], bbox[2], bbox[3]
                                    object_visual = image.crop((left, top, left+width, top+height))

                                    self.post_input[cnt] = {}
                                    self.post_input[cnt]['visual'] = object_visual
                                    self.post_input[cnt]['visual_meta'] = visual_meta_flatten
                                    self.post_input[cnt]['dialog_cnt'] = dialog_cnt
                                    self.dial2object[dialog_cnt].append(visual_meta_flatten)
                                    cnt += 1
                                    
        for key, data in self.post_input.items():
            dialog_cnt = data['dialog_cnt']
            visual_meta = data['visual_meta']
            dial_objects = self.dial2object[dialog_cnt]
            self.post_input[key]['neg_visual_meta'] = []

            for dial_object in dial_objects:
                if dial_object != visual_meta:
                    self.post_input[key]['neg_visual_meta'].append(dial_object)
            
    def __len__(self):
        return len(self.post_input)

    def __getitem__(self, idx):
        return self.post_input[idx]    