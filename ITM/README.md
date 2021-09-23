# Post-training

Before fine-tuning each task, we perform post-training that fuses the text modal base model and the vision modal model. This step is one of image-text-matching, where images are objects in a scene and texts are visual metadata of objects. This can make the representation of an object similar to the representation vector of visual metadata by comparing the representation between text and image. Negative data are all other objects in the same session except for the positive object. This improves the performance of our model.

## Download the pre-trained model 
Download model.pt from the [post_training/all folder](https://github.com/rungjoo/dstc10/tree/master/ITM/post_training/all).

## Training
```bash
bash train.sh
```
The model and log (train.log) is saved in the post_training/all folder.