# The Submission for SIMMC 2.0 Challenge 2021
- [challenge website](https://github.com/facebookresearch/simmc2)

## Requirements
- python 3.8.8
- pytorch 1.8.1
- [transformers 4.8.2](https://huggingface.co/transformers/v4.8.1/)
- [apex](https://github.com/NVIDIA/apex) for multi-gpu
- nltk

## Preprocessing

1. Download Data

Download the [data](https://github.com/facebookresearch/simmc2/tree/master/data) provided by the challenge organizer and put it in the data folder.

2. Image saving

Preprocess the image files in advance. The preprocessed result has the image name as the key and visual as the value.
```bash
python3 image_preprocessor.py
```
The result is saved in res folder.

## Step 1

First, the model is post-trained by image-to-text matching. Here, image is each object and text is the visual metadata of the image.
Code is provided in the ITM folder.

## Step 2

This is the learning process for each subtask. You can train the model in each folder (sub1, sub2, sub4).