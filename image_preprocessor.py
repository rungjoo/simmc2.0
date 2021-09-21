import os
import glob
import argparse
import pickle
import pdb

from PIL import Image, ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

def process_images(image_path1, image_path2, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_list1 = glob.glob(image_path1) # "{YOUR_DATA_PATH}/simmc2_scene_images_dstc10_public_part1/*"
    image_list2 = glob.glob(image_path2) # "{YOUR_DATA_PATH}/simmc2_scene_images_dstc10_public_part2/*"

    image_list = image_list1+image_list2

    # Preload image and save as pickle
    image_visual = {}
    for idx, image_path in tqdm(enumerate(image_list), total=len(image_list), desc="Loading Images"):
        img_key = os.path.splitext(os.path.basename(image_path))[0]
        try:
            image_visual[img_key] = Image.open(image_path).convert('RGB')
        except:
            print("ERROR", image_path)

    print ("Saving...") 
    output_path = os.path.join(output_dir, 'image_obj.pickle') # image_obj.pickle
    with open(output_path, 'wb') as f:
        pickle.dump(image_visual, f, pickle.HIGHEST_PROTOCOL)
    print ("Finished")


def main(args):
    process_images(args.input1, args.input2, args.output_dir)

if __name__ == '__main__': 
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Image data save" )
    parser.add_argument( "--input1", type=str, default="./data/simmc2_scene_images_dstc10_public_part1/*")
    parser.add_argument( "--input2", type=str, default="./data/simmc2_scene_images_dstc10_public_part2/*")
    parser.add_argument( "--output_dir", type=str,  default='./res/')
       
    args = parser.parse_args()
     
    main(args)
