from torch.utils.data import Dataset, DataLoader

import random, pdb
from tqdm import tqdm

import json
import glob, os
import pickle
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# import matplotlib.pyplot as plt
        
class task2_loader(Dataset):
    def __init__(self, data_path, image_obj_path, description_path, fashion_path, furniture_path):
        with open(data_path, 'r') as f: # './simmc2/data/simmc2_dials_dstc10_train.json'
            json_data = json.load(f)
            dialogue_data = json_data['dialogue_data']

        """ 이미지 입력 """
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
        
        self.task2_input = {}
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

            """ 이미지 설명 저장 """
            self.dial2object[dialog_cnt] = {}            
            self.dial2rel[dialog_cnt] = {}
            # self.dial2bg[dialog_cnt] = {}
            self.dial2object[dialog_cnt]['object'] = {}            
            total_objects = []
            
            for k, image_name in scene_ids.items():
                if image_name[:2] == 'm_':
                    image_find_name = image_name[2:]
                else:
                    image_find_name = image_name
                image = image_visual[image_find_name]
                
                for image_des_path in image_des_list:
                    if image_name in image_des_path: # 사용하는 이미지 찾기
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

                                    ## object 2D & meta 저장
                                    visual_metalist = []
                                    non_visual_metalist = []
                                    for k, v in metadata[prefab_path].items():
                                        if k in visual_meta_type:
                                            visual_metalist.append(v)
                                        elif k in non_visual_meta_type:
                                            non_visual_metalist.append(v)
                                    visual_meta_flatten = ' '.join([str(x) for x in visual_metalist])
                                    non_visual_meta_flatten = ' '.join([str(x) for x in non_visual_metalist])

                                    if object_id not in self.dial2object[dialog_cnt]['object'].keys():
                                        self.dial2object[dialog_cnt]['object'][object_id] = {}
                                        self.dial2object[dialog_cnt]['object'][object_id]['visual_meta'] = visual_meta_flatten
                                        self.dial2object[dialog_cnt]['object'][object_id]['non_visual_meta'] = non_visual_meta_flatten
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
            cand_objects = list(set(total_objects))
                
                                        
            """ 텍스트 입력 """
            text_data = one_dialogue['dialogue']
            
            system2object_list = []
            system2object_id_list = []
            utt2object_list = {}
            task2_sample_input = ''
            for i, text in enumerate(text_data):
                """ user 텍스트 입력 """
                transcript = text['transcript']        
                if i == 0:
                    task2_sample_input += '[USER] '
                else:
                    task2_sample_input += ' [USER] '                
                task2_sample_input += transcript

                transcript_annotated = text['transcript_annotated']
                transcript_objects = transcript_annotated['act_attributes']['objects']
                
                """ for system matching """
                for object_id in cand_objects:
                    if object_id in transcript_objects:                    
                        utt2object_list[object_id] = 1
                    else:
                        utt2object_list[object_id] = 0
                system2label, objcat2label = self.utt2system(utt2object_list, system2object_list[:])
                
                """ data save """
                # for obj_cnt, (object_id, dial2object_data) in enumerate(self.dial2object[dialog_cnt]['object'].items()):     
                for obj_cnt, (object_id) in enumerate(cand_objects):
                    dial2object_data = self.dial2object[dialog_cnt]['object'][object_id]
                    self.task2_input[cnt] = {}
                    self.task2_input[cnt]['input'] = task2_sample_input
                    self.task2_input[cnt]['object_id'] = object_id
                    self.task2_input[cnt]['sess_cnt'] = i
                    if object_id in transcript_objects:                 
                        self.task2_input[cnt]['object_label'] = 1
                    else:
                        self.task2_input[cnt]['object_label'] = 0
                    # self.task2_input[cnt]['dial2rel'] = self.dial2rel[dialog_cnt]
                    if obj_cnt == 0:
                        self.task2_input[cnt]['system_label'] = system2label
                        self.task2_input[cnt]['uttcat_label'] = objcat2label
                        self.task2_input[cnt]['pre_system_objects'] = system2object_id_list[:]
                    else:
                        self.task2_input[cnt]['system_label'] = system2label # 학습에 사용 X
                        self.task2_input[cnt]['uttcat_label'] = -100 # objcat2label # 학습에 사용 X
                        self.task2_input[cnt]['pre_system_objects'] = system2object_id_list[:]
                    self.task2_input[cnt]['visual'] = dial2object_data['visual']
                    self.task2_input[cnt]['visual_meta'] = dial2object_data['visual_meta']
                    self.task2_input[cnt]['background'] = dial2object_data['background']
                    cnt += 1

                """ system 텍스트 입력 """
                system_transcript = text['system_transcript']
                task2_sample_input += ' [SYSTEM] '
                task2_sample_input += system_transcript
                
                system_transcript_annotated = text['system_transcript_annotated']
                system_transcript_objects = system_transcript_annotated['act_attributes']['objects']
                
                ## system object matching
                system2object = {}
                system2object_id = []
                for object_id in cand_objects:
                    if object_id in system_transcript_objects:
                        system2object[object_id] = 1
                        system2object_id.append(object_id)
                    else:
                        system2object[object_id] = 0
                system2object_list.append(system2object)
                system2object_id_list.append(system2object_id)                
                

    def utt2system(self, utt_obj, sys_obj_list):
        utt_match_id = []
        for utt_obj_id, utt_label in utt_obj.items():
            if utt_label == 1:
                utt_match_id.append(utt_obj_id)
        
        """ system matching label (0,1) """
        system2label = []
        total_sys_match_id = []
        for turn, sys_obj in enumerate(sys_obj_list):
            sys_match_id = []
            for sys_obj_id, sys_label in sys_obj.items():
                if sys_label == 1:
                    sys_match_id.append(sys_obj_id)
                    total_sys_match_id.append(sys_obj_id)
            
            temp_match_id = []
            match_num = 0
            for sys_obj_id in sys_match_id:
                if sys_obj_id in utt_match_id:
                    match_num += 1
                    temp_match_id.append(sys_obj_id)
            
            if match_num == 0:
                system2label.append(0)
            else:
                system2label.append(1)
        
        """ object category label (0,1,2,3) """
        objcat2label = []        
        match_num = 0
        non_match_num = 0
        for utt_obj_id in utt_match_id:
            if utt_obj_id in total_sys_match_id:
                match_num += 1
            else:
                non_match_num += 1
        
        if match_num == 0:
            if non_match_num == 0:
                objcat2label = 0 # 매칭될 object 존재가 없는 발화
            else:
                objcat2label = 1 # 이전의 system에서 언급된 object는 없고, 새로운 object가 있는 것
        else:
            if non_match_num == 0:
                objcat2label = 1 # 2 이전의 system에서 언급된 object들이 후보일 경우
            else:
                objcat2label = 1 # 3 이전의 system에서 언급된 object들도 있고 새로운 object도 후보일 경우        
        
        return system2label, objcat2label
                
    def __len__(self):
        return len(self.task2_input)

    def __getitem__(self, idx):
        return self.task2_input[idx]