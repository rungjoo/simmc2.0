# The Submission for SIMMC 2.0 Challenge 2021
- [challenge website](https://github.com/facebookresearch/simmc2)

## Requirements
- python 3.7
- pytorch 1.8.1
- [transformers 4.8.2](https://huggingface.co/transformers/v4.8.1/)
- [apex](https://github.com/NVIDIA/apex) for multi-gpu

## Step 1

First, the model is post-trained by image-to-text matching. Here, image is each object and text is the visual metadata of the image.
Code is provided in the ITM folder.