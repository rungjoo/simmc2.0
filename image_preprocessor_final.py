import os
import glob
import argparse
import pickle
import pdb

from PIL import Image, ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_images(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_list = glob.glob(image_path) # "{YOUR_DATA_PATH}/simmc2_scene_images_dstc10_teststd/*"

    # Preload image and save as pickle
    image_visual = {}
    for idx, image_path in tqdm(enumerate(image_list), total=len(image_list), desc="Loading Images"):
        img_key = os.path.splitext(os.path.basename(image_path))[0]
        try:
            image_visual[img_key] = Image.open(image_path).convert('RGB')
        except:
            print("ERROR", image_path)

    print ("Saving...") 
    output_path = os.path.join(output_dir, 'image_obj_final.pickle') # image_obj.pickle
    with open(output_path, 'wb') as f:
        pickle.dump(image_visual, f, pickle.HIGHEST_PROTOCOL)
    print ("Finished")


def main(args):
    process_images(args.input, args.output_dir)

if __name__ == '__main__': 
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Image data save" )
    parser.add_argument( "--input", type=str, default="./data/simmc2_scene_images_dstc10_teststd/*")
    parser.add_argument( "--output_dir", type=str,  default='./res/')
       
    args = parser.parse_args()
     
    main(args)
